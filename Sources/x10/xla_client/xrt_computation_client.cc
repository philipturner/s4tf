// Copyright 2020 TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"

#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <list>
#include <sstream>
#include <unordered_map>

#include "absl/container/node_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/xla_client/env_vars.h"
#include "tensorflow/compiler/xla/xla_client/multi_wait.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/thread_pool.h"
#include "tensorflow/compiler/xla/xla_client/unique.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"
#include "tensorflow/compiler/xla/xla_client/xrt_local_service.h"
#include "tensorflow/compiler/xla/xla_client/local_device.h"
#include "tensorflow/compiler/xrt/xrt_util.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/net.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace xla {

using DataPtr = ComputationClient::DataPtr;
using ComputationPtr = ComputationClient::ComputationPtr;
using TensorSource = ComputationClient::TensorSource;

class XrtComputationClient::XrtDevice : public ComputationClient::Device {
 public:
  XrtDevice(std::string name, XrtComputationClient* client)
      : Device(name), client_(client) {}

  DataPtr CreateDataPlaceholder(Shape shape) override {
    return std::make_shared<XrtData>(this, std::move(shape));
  }

  std::vector<DataPtr> TransferToServer(
      absl::Span<const TensorSource> tensors) override;

  std::vector<ComputationClient::ComputationPtr> Compile(
      const std::vector<std::string>& devices,
      std::vector<CompileInstance> instances) override {
    return client_->Compile(name(), devices, std::move(instances));
  }

  std::string ResourceDomain() const override {
    return client_->GetResourceDomain(name());
  }

  std::vector<ComputationClient::DataPtr> ExecuteChained(
      absl::Span<const ComputationClient::ExecuteChainedOp> ops) override {
    return client_->ExecuteChained(ops, name());
  }

  std::vector<ComputationClient::DataPtr> ExecuteComputation(
      const Computation& computation, absl::Span<const DataPtr> arguments,
      const ExecuteComputationOptions& options) override {
    return client_->ExecuteComputation(computation, arguments, name(), options);
  }

  ComputationClient::TransferManager* GetTransferManager() const override {
    return client_;
  }

  XrtComputationClient* computation_client() const { return client_; }

 private:
  XrtComputationClient* client_;
};

XrtComputationClient::XrtData::XrtData(XrtDevice* device, Shape device_shape,
                                       int64_t handle)
    : Data(device, std::move(device_shape)),
      handle_ptr(std::make_shared<XrtHandle>(handle, [device, handle]() {
        reinterpret_cast<XrtComputationClient*>(device->computation_client())
            ->ReleaseXrtData(device->name(), handle);
      })) {}

namespace {

struct DeviceCountDefaults {
  int num_tpus = 0;
  int num_gpus = 0;
  int num_cpus = 1;
};

static const char* const kLocalService = "localservice";

struct TensorAllocatorTraits {
  static void *allocate(size_t size, size_t alignment) {
#if defined(_WIN32)
    return ::_aligned_malloc(alignment, size);
#elif defined(__APPLE__)
    void *ptr;
    ::posix_memalign(&ptr, alignment, size);
    return ptr;
#else
    return ::aligned_alloc(alignment, size);
#endif
  }

  static void deallocate(void *allocation) {
#if defined(_WIN32)
    return ::_aligned_free(allocation);
#else
    return ::free(allocation);
#endif
  }
};

// A simple Tensorflow Allocator which caches Tensor allocations in order to
// avoid paying the kernel's clear_page_c() price.
class TensorAllocator : public tensorflow::Allocator {
  struct AllocKey {
    struct Hash {
      size_t operator()(const AllocKey& hk) const {
        return util::StdHashCombine(hk.alignment, hk.num_bytes);
      }
    };

    bool operator==(const AllocKey& rhs) const {
      return num_bytes == rhs.num_bytes && alignment == rhs.alignment;
    }

    size_t alignment = 0;
    size_t num_bytes = 0;
  };

  struct AllocBlocks {
    explicit AllocBlocks(const AllocKey& alloc_key) : alloc_key(alloc_key) {}

    AllocKey alloc_key;
    std::vector<void*> blocks;
  };

  using AllocList = std::list<AllocBlocks>;

 public:
  static TensorAllocator* Get() {
    static size_t max_size =
        sys_util::GetEnvInt("XLA_TENSOR_ALLOCATOR_MAXSIZE", 1000000000);
    static TensorAllocator* allocator = new TensorAllocator(max_size);
    return allocator;
  }

  std::string Name() override { return "XLA_TensorAllocator"; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    // We use an alignment-sized area before the memory returned to the caller,
    // to store a pointer to its AllocBlocks.
    alignment = std::max<size_t>(alignment, sizeof(void*));
    // To call aligned_alloc(), num_bytes must be multiple of alignment.
    num_bytes = RoundUpTo(num_bytes, alignment);

    AllocKey alloc_key = {alignment, num_bytes};
    void* block = nullptr;
    AllocBlocks* alloc_blocks = nullptr;
    std::lock_guard<std::mutex> lock(lock_);
    auto it = allocs_.find(alloc_key);
    if (it != allocs_.end()) {
      alloc_blocks = &*it->second;
      if (!alloc_blocks->blocks.empty()) {
        block = alloc_blocks->blocks.back();
        alloc_blocks->blocks.pop_back();
      }
      // LRU
      alloc_list_.splice(alloc_list_.begin(), alloc_list_, it->second);
    } else {
      allocs_.emplace(alloc_key, alloc_list_.insert(alloc_list_.begin(),
                                                    AllocBlocks(alloc_key)));
      alloc_blocks = &alloc_list_.front();
    }
    if (block == nullptr) {
      TrimCache(alloc_key.num_bytes);
      block = NewBlock(alloc_blocks);
    }
    return block;
  }

  void DeallocateRaw(void* ptr) override {
    if (ptr != nullptr) {
      // The pointer to AllocBlocks is right before the user memory.
      AllocBlocks* alloc_blocks = reinterpret_cast<AllocBlocks**>(ptr)[-1];
      std::lock_guard<std::mutex> lock(lock_);
      if (alloc_blocks->alloc_key.num_bytes < max_size_) {
        alloc_blocks->blocks.push_back(ptr);
      } else {
        // We do not cache blocks whose size is bigger than the max cache size.
        FreeBlock(ptr, alloc_blocks);
      }
    }
  }

 private:
  explicit TensorAllocator(size_t max_size) : max_size_(max_size) {}

  void* NewBlock(AllocBlocks* alloc_blocks) {
    // We allocate an extra alignment sized area to store the AllocBlocks
    // pointer.
    void *ptr = TensorAllocatorTraits::allocate(
        alloc_blocks->alloc_key.alignment + alloc_blocks->alloc_key.num_bytes,
        alloc_blocks->alloc_key.alignment);
    XLA_CHECK(ptr != nullptr);
    ptr = reinterpret_cast<char*>(ptr) + alloc_blocks->alloc_key.alignment;
    // Store the pointer to AllocBlocks right before the user memory.
    reinterpret_cast<AllocBlocks**>(ptr)[-1] = alloc_blocks;
    size_ += alloc_blocks->alloc_key.num_bytes;
    return ptr;
  }

  void FreeBlock(void* ptr, AllocBlocks* alloc_blocks) {
    size_ -= alloc_blocks->alloc_key.num_bytes;
    TensorAllocatorTraits::deallocate(
        reinterpret_cast<char*>(ptr) - alloc_blocks->alloc_key.alignment);
  }

  void TrimCache(size_t num_bytes) {
    auto it = alloc_list_.rbegin();
    for (; size_ + num_bytes > max_size_ && it != alloc_list_.rend(); ++it) {
      AllocBlocks* alloc_blocks = &*it;
      while (!alloc_blocks->blocks.empty() && size_ + num_bytes > max_size_) {
        FreeBlock(alloc_blocks->blocks.back(), alloc_blocks);
        alloc_blocks->blocks.pop_back();
      }
    }
  }

  size_t max_size_ = 0;
  std::mutex lock_;
  size_t size_ = 0;
  AllocList alloc_list_;
  absl::node_hash_map<AllocKey, AllocList::iterator, AllocKey::Hash> allocs_;
};

std::string StripPrefix(const std::string& value, const std::string& prefix) {
  return absl::StartsWith(value, prefix) ? value.substr(prefix.size()) : value;
}

tensorflow::DeviceNameUtils::ParsedName ParseFullXrtDevice(
    const std::string& device) {
  tensorflow::DeviceNameUtils::ParsedName parsed_device;
  XLA_CHECK(
      tensorflow::DeviceNameUtils::ParseFullName(device, &parsed_device) &&
      parsed_device.has_job && parsed_device.has_task && parsed_device.has_id &&
      parsed_device.has_type)
      << device;
  return parsed_device;
}

void MaybeSaveLongCompileHlo(double compile_time,
                             const XlaComputation& computation) {
  static double compile_time_threshold = sys_util::GetEnvDouble(
      "XLA_COMPILE_TIME_THRESHOLD", std::numeric_limits<double>::max());
  static const std::string* hlo_folder = new std::string(
      sys_util::GetEnvString("XLA_SLOW_COMPILE_HLO_FOLDER", ""));
  if (compile_time > compile_time_threshold && !hlo_folder->empty()) {
    static std::atomic<size_t> hlo_count(0);
    std::stringstream ss;
    ss << *hlo_folder << "/hlo_module-" << hlo_count.fetch_add(1) << "-"
       << static_cast<int64_t>(compile_time) << "s.txt";
    std::string hlo_text =
        ConsumeValue(util::GetComputationHloText(computation));
    std::ofstream graph_file(ss.str());
    graph_file << hlo_text << "\n";
  }
}

std::string MakeGrpcEndPoint(const std::string& server) {
  return server.compare(0, 7, "grpc://") == 0 ? server
                                              : absl::StrCat("grpc://", server);
}

std::string GetXrtDevicePath(const std::string& worker, int task_no,
                             const std::string& device_type, int ordinal) {
  return absl::StrCat("/job:", worker, "/replica:0/task:", task_no,
                      "/device:", device_type, ":", ordinal);
}

std::string BuildTaskDeviceKey(int task_no, const std::string& kind) {
  return absl::StrCat(task_no, ":", kind);
}

tensorflow::DeviceNameUtils::ParsedName ParseXrtDevice(
    const std::string& device) {
  tensorflow::DeviceNameUtils::ParsedName parsed_device;
  XLA_CHECK(
      tensorflow::DeviceNameUtils::ParseFullName(device, &parsed_device) &&
      parsed_device.has_job && parsed_device.has_task && parsed_device.has_id &&
      parsed_device.has_type)
      << device;
  return parsed_device;
}

bool IsLocalDevice(const XrtComputationClient::Worker& worker,
                   const tensorflow::DeviceNameUtils::ParsedName& parsed_device,
                   const std::map<std::string, int>& dev_task_map) {
  if (worker.name != parsed_device.job ||
      worker.task_no != parsed_device.task) {
    return false;
  }
  std::string mp_device = XrtComputationClient::GetMultiProcessingDevice();
  if (mp_device.empty()) {
    return true;
  }
  XrtComputationClient::DeviceId device(mp_device);
  std::string task_device_key =
      BuildTaskDeviceKey(parsed_device.task, device.kind);
  auto it = dev_task_map.find(task_device_key);
  return it != dev_task_map.end()
             ? (device.ordinal == it->second + parsed_device.id)
             : false;
}

std::map<std::string, int> BuildDeviceTaskMap(
    const XrtComputationClient::Options& options) {
  // Builds a map from "TASK:DEV_KIND" (ie, "0:TPU") keys to the minimum global
  // device ordinal assigned for that task+devkind couple.
  std::map<std::string, int> dev_task_map;
  for (auto& device_xrt_device : options.global_device_map) {
    XrtComputationClient::DeviceId global_device(device_xrt_device.first);
    tensorflow::DeviceNameUtils::ParsedName parsed_device =
        ParseXrtDevice(device_xrt_device.second);
    std::string task_device_key =
        BuildTaskDeviceKey(parsed_device.task, global_device.kind);
    util::InsertCombined(&dev_task_map, task_device_key, global_device.ordinal,
                         [](int a, int b) { return std::min(a, b); });
  }
  return dev_task_map;
}

void PopulateLocalDevices(XrtComputationClient::Options* options) {
  std::string local_worker = sys_util::GetEnvString(env::kEnvLocalWorker, "");
  XrtComputationClient::Worker worker("", -1);
  if (!local_worker.empty()) {
    worker = XrtComputationClient::ParseWorker(local_worker);
  }
  auto dev_task_map = BuildDeviceTaskMap(*options);
  std::map<std::string, int> min_ordinals;
  for (auto& device_xrt_device : options->global_device_map) {
    if (worker.task_no >= 0) {
      tensorflow::DeviceNameUtils::ParsedName parsed_device =
          ParseXrtDevice(device_xrt_device.second);
      if (!IsLocalDevice(worker, parsed_device, dev_task_map)) {
        continue;
      }
    }
    options->devices.insert(device_xrt_device.first);

    XrtComputationClient::DeviceId global_device(device_xrt_device.first);
    util::InsertCombined(&min_ordinals, global_device.kind,
                         global_device.ordinal,
                         [](int a, int b) { return std::min(a, b); });
  }
  for (auto kind : {"TPU", "GPU", "CPU"}) {
    auto it = min_ordinals.find(kind);
    if (it != min_ordinals.end()) {
      options->default_device = absl::StrCat(kind, ":", it->second);
      break;
    }
  }
}

void AddXrtHostDevices(const std::string& worker_name, int task_no,
                       const std::string& server,
                       const DeviceCountDefaults& device_counts,
                       std::map<std::string, int>* device_ordinals,
                       XrtComputationClient::Options* options) {
  struct Devices {
    const char* name;
    const char* tf_name;
    int count;
  } const devices[] = {
      {"TPU", "TPU",
       static_cast<int>(
           sys_util::GetEnvInt(env::kEnvNumTpu, device_counts.num_tpus))},
      {"GPU", "XLA_GPU",
       static_cast<int>(
           sys_util::GetEnvInt(env::kEnvNumGpu, device_counts.num_gpus))},
      {"CPU", "XLA_CPU",
       static_cast<int>(
           sys_util::GetEnvInt(env::kEnvNumCpu, device_counts.num_cpus))},
  };
  options->workers_map.emplace(
      XrtComputationClient::Worker(worker_name, task_no),
      MakeGrpcEndPoint(server));
  for (auto& device : devices) {
    int& device_ordinal = (*device_ordinals)[device.name];
    for (int j = 0; j < device.count; ++j, ++device_ordinal) {
      std::string device_name = absl::StrCat(device.name, ":", device_ordinal);
      std::string xrt_device_name =
          GetXrtDevicePath(worker_name, task_no, device.tf_name, j);
      options->global_device_map.emplace(device_name, xrt_device_name);
    }
  }
}

bool ParseEnvBasedTpuClusterConfig(XrtComputationClient::Options* options) {
  std::string tpu_config = sys_util::GetEnvString(env::kEnvTpuConfig, "");
  if (tpu_config.empty()) {
    return false;
  }
  std::map<std::string, int> device_ordinals;
  std::vector<std::string> spec_parts = absl::StrSplit(tpu_config, '|');
  XLA_CHECK(!spec_parts.empty()) << tpu_config;
  DeviceCountDefaults device_counts;
  device_counts.num_tpus = 8;
  for (const auto& spec : spec_parts) {
    std::vector<std::string> host_parts = absl::StrSplit(spec, ';');
    XLA_CHECK_EQ(host_parts.size(), 3) << spec;
    AddXrtHostDevices(host_parts[0], std::stoi(host_parts[1]), host_parts[2],
                      device_counts, &device_ordinals, options);
  }
  return true;
}

bool ParseMeshConfig(
    XrtComputationClient::Options* options,
    std::unique_ptr<tensorflow::tpu::TopologyProto>* topology_proto) {
  service::MeshClient* client = service::MeshClient::Get();
  if (client == nullptr) {
    return false;
  }
  std::string local_worker_env =
      sys_util::GetEnvString(env::kEnvLocalWorker, "");
  XLA_CHECK(!local_worker_env.empty())
      << "In a mesh client setup the XRT_LOCAL_WORKER must be specified";

  XrtComputationClient::Worker local_worker =
      XrtComputationClient::ParseWorker(local_worker_env);

  TF_LOG(INFO) << "Fetching mesh configuration for worker " << local_worker.name
               << ":" << local_worker.task_no << " from mesh service at "
               << client->address();
  service::grpc::Config config = client->GetConfig();
  TF_VLOG(3) << "Mesh Config: " << config.DebugString();

  std::string mp_device = XrtComputationClient::GetMultiProcessingDevice();
  for (auto& config_worker : config.workers()) {
    XrtComputationClient::Worker worker(config_worker.name(),
                                        config_worker.task_no());
    options->workers_map.emplace(worker, config_worker.address());

    for (auto& device : config_worker.devices()) {
      XrtComputationClient::DeviceId local_device(device.local_name());
      options->global_device_map.emplace(
          device.global_name(),
          GetXrtDevicePath(worker.name, worker.task_no, local_device.kind,
                           local_device.ordinal));
      if (local_worker == worker &&
          (mp_device.empty() || device.global_name() == mp_device)) {
        options->devices.insert(device.global_name());
      }
    }
  }
  (*topology_proto) = absl::make_unique<tensorflow::tpu::TopologyProto>(
      std::move(*config.mutable_proto()));
  return true;
}

template <typename T>
T ParseProto(const tensorflow::Tensor& tensor) {
  const tensorflow::tstring& tensor_data =
      tensor.scalar<tensorflow::tstring>()();
  // The ParseFromArray() API takes an 'int' as size argument, so the tensor
  // size better be fitting the 'int' domain.
  XLA_CHECK_LE(tensor_data.size(),
               static_cast<size_t>(std::numeric_limits<int>::max()));
  T proto;
  XLA_CHECK(proto.ParseFromArray(tensor_data.data(), tensor_data.size()));
  return proto;
}

int64_t GetMaxTensorsPartitionSize() {
  // We need to limit the amount of data we send to the XRT backend since
  // Protocol Buffers does not allow sizes greater than 2GB. We keep some margin
  // to avoid extra metadata pushing us over the limit.
  static int64_t max_partition_size =
      sys_util::GetEnvInt("XRT_MAX_TENSORS_PARTITION", 1800000000);
  return max_partition_size;
}

bool GpuIsAvailable() {
  std::vector<string> devices;
  tensorflow::Status s =
      tensorflow::DeviceFactory::ListAllPhysicalDevices(&devices);
  XLA_CHECK_OK(s);
  for (const std::string& device : devices) {
    std::vector<std::string> device_parts = absl::StrSplit(device, ':');
    XLA_CHECK_EQ(device_parts.size(), 3) << device;
    if (device_parts[1] == "GPU") {
      return true;
    }
  }
  return false;
}

bool ParseEnvDeviceCounts(XrtComputationClient::Options* options) {
  int num_tpus = sys_util::GetEnvInt(env::kEnvNumTpu, -1);
  int num_gpus = sys_util::GetEnvInt(env::kEnvNumGpu, -1);
  if (num_tpus > 0 || num_gpus > 0) {
    std::map<std::string, int> device_ordinals;
    std::string host_port =
        absl::StrCat("localhost:", tensorflow::internal::PickUnusedPortOrDie());
    AddXrtHostDevices("localservice", 0, host_port, DeviceCountDefaults(),
                      &device_ordinals, options);
  }
  return !options->global_device_map.empty();
}

bool ParseEnvDevices(XrtComputationClient::Options* options) {
  std::string device = "CPU";
  std::string default_device_spec = absl::StrFormat(
      "%s:0;/job:localservice/replica:0/task:0/device:XLA_%s:0", device,
      device);
  std::string device_spec =
      sys_util::GetEnvString(env::kEnvDeviceMap, default_device_spec);
  int port = tensorflow::internal::PickUnusedPortOrDie();
  std::string workers_spec = sys_util::GetEnvString(
      env::kEnvWorkers, absl::StrCat("localservice:0;grpc://localhost:", port));
  if (!device_spec.empty() && !workers_spec.empty()) {
    for (const auto& device_target : absl::StrSplit(device_spec, '|')) {
      std::vector<std::string> parts = absl::StrSplit(device_target, ';');
      XLA_CHECK_EQ(parts.size(), 2) << device_target;
      options->global_device_map.emplace(parts[0], parts[1]);
    }
    for (const auto& name_target : absl::StrSplit(workers_spec, '|')) {
      std::vector<std::string> parts = absl::StrSplit(name_target, ';');
      XLA_CHECK_EQ(parts.size(), 2) << name_target;
      options->workers_map.emplace(XrtComputationClient::ParseWorker(parts[0]),
                                   MakeGrpcEndPoint(parts[1]));
    }
  }
  return !options->global_device_map.empty();
}

}  // namespace

std::unique_ptr<ComputationClient> ComputationClient::Create() {
  XrtComputationClient::Options options;
  std::unique_ptr<tensorflow::tpu::TopologyProto> topology_proto;
  if (!ParseEnvBasedTpuClusterConfig(&options) &&
      !ParseEnvDeviceCounts(&options) && !ParseEnvDevices(&options) &&
      !ParseMeshConfig(&options, &topology_proto)) {
    XLA_ERROR() << "Missing XLA configuration";
  }
  PopulateLocalDevices(&options);
  return std::unique_ptr<ComputationClient>(
      new XrtComputationClient(options, std::move(topology_proto)));
}

XrtComputationClient::DeviceId::DeviceId(const std::string& device_str) {
  std::vector<std::string> parts = absl::StrSplit(device_str, ':');
  XLA_CHECK_EQ(parts.size(), 2) << device_str;
  kind = std::move(parts[0]);
  ordinal = std::stoi(parts[1]);
}

void XrtComputationClient::XrtData::Assign(const Data& data) {
  const XrtData& xrt_data = dynamic_cast<const XrtData&>(data);
  if (&xrt_data != this) {
    handle_ptr = xrt_data.handle_ptr;
  }
}

XrtComputationClient::XrtComputationClient(
    Options options,
    std::unique_ptr<tensorflow::tpu::TopologyProto> topology_proto)
    : options_(std::move(options)),
      compilation_cache_(sys_util::GetEnvInt("XLA_COMPILATION_CACHE_SIZE", 64)),
      rng_seed_(0x5a2d296e9) {
  tensorflow::ConfigProto config = CreateConfigProto(options_);
  std::string local_target = GetLocalTarget(options_);
  session_cache_ = absl::make_unique<XrtSessionCache>(
      config, [this](XrtSession* s) { InitSession(s); }, local_target);
  alloc_session_cache_ =
      absl::make_unique<XrtSessionCache>(config, nullptr, local_target);

  auto default_device_target =
      options_.global_device_map.find(options_.default_device);
  XLA_CHECK(default_device_target != options_.global_device_map.end())
      << options_.default_device;
  for (auto& device : options_.devices) {
    XLA_CHECK(options_.global_device_map.find(device) !=
              options_.global_device_map.end())
        << "Missing device in global map: " << device;
  }
  for (const auto& dev_target : options_.global_device_map) {
    const char* tag =
        options_.devices.count(dev_target.first) > 0 ? "LOCAL" : "REMOTE";
    TF_VLOG(1) << "XRT device (" << tag << ") " << dev_target.first << " -> "
               << dev_target.second;
  }
  for (auto& worker_target : options_.workers_map) {
    TF_VLOG(1) << "Worker " << worker_target.second
               << " for /job:" << worker_target.first.name
               << "/replica:0/task:" << worker_target.first.task_no;
  }
  TF_VLOG(1) << "XRT default device: " << options_.default_device;
  MaybeCreateLocalService(options_);
  InitializeDevices(std::move(topology_proto));
  StartHandleReleaser();

  for (const auto& dev_target : options_.global_device_map) {
    AddDevice(std::make_unique<XrtDevice>(dev_target.first, this));
  }

  for (auto& device : GetAllLocalDevicesForPlatform("gpu", "GPU")) {
    options_.default_device = "GPU:0";
    AddDevice(std::move(device));
  }
}

std::vector<size_t> XrtComputationClient::PartitionTransferToServer(
    absl::Span<const TensorSource> tensors) {
  int64_t max_partition_size = GetMaxTensorsPartitionSize();
  uint64 current_size = 0;
  std::vector<size_t> partitions;
  for (size_t i = 0; i < tensors.size(); ++i) {
    int64_t tensor_size = ShapeUtil::ByteSizeOfElements(tensors[i].shape);
    if (current_size + tensor_size > max_partition_size) {
      if (partitions.empty() && i > 0) {
        partitions.push_back(0);
      }
      partitions.push_back(i);
      current_size = 0;
    }
    current_size += tensor_size;
  }
  if (partitions.empty()) {
    partitions.push_back(0);
  }
  return partitions;
}

std::vector<ComputationClient::DataPtr>
XrtComputationClient::XrtDevice::TransferToServer(
    absl::Span<const TensorSource> tensors) {
  auto partitions = PartitionTransferToServer(tensors);
  if (partitions.size() == 1) {
    // Fast path in case of single partition. Avoid creating threads and
    // waiting, since this is the common case.
    return client_->TransferToServerInternal(this, tensors);
  }
  XLA_COUNTER("XrtPartitionedTransferToServer", 1);

  util::MultiWait mwait(partitions.size());
  std::vector<DataPtr> results(tensors.size());
  for (size_t i = 0; i < partitions.size(); ++i) {
    auto sender = [&, i]() {
      size_t base_index = partitions[i];
      size_t length = (i + 1 < partitions.size())
                          ? partitions[i + 1] - base_index
                          : tensors.size() - base_index;
      auto partitions_results = client_->TransferToServerInternal(
          this, tensors.subspan(base_index, length));
      for (size_t r = 0; r < length; ++r) {
        results[base_index + r] = std::move(partitions_results[r]);
      }
    };
    env::ScheduleIoClosure(mwait.Completer(std::move(sender)));
  }
  mwait.Wait();
  return results;
}

std::vector<ComputationClient::DataPtr>
XrtComputationClient::TransferToServerInternal(
    XrtDevice* device_ptr, absl::Span<const TensorSource> tensors) {
  metrics::TimedSection timed(TransferToServerMetric());

  std::mutex lock;
  XrtSessionCache::SessionMap session_map;
  int64_t total_size = 0;
  util::MultiWait mwait(tensors.size());
  std::map<XrtSession*, SessionWork> session_work_map;
  std::string device = GetEffectiveDevice(device_ptr->name());
  {
    metrics::TimedSection timed(TransferToServerTransformMetric());

    for (size_t i = 0; i < tensors.size(); ++i) {
      auto converter = [&, i]() {
        const std::string& xrt_device = SwiftDeviceToXrtDevice(device);
        tensorflow::Tensor tensor(
            TensorAllocator::Get(),
            XlaTypeToDataType(tensors[i].shape.element_type()),
            MakeEquivalentTensorShape(tensors[i].shape));
        auto tdata = tensor.tensor_data();
        tensors[i].populate_fn(tensors[i], const_cast<char*>(tdata.data()),
                               tdata.size());

        {
          std::lock_guard<std::mutex> slock(lock);
          XrtSession* session = GetSessionForXrtDevice(
              alloc_session_cache_.get(), xrt_device, &session_map);
          SessionWork* session_work = &session_work_map[session];
          tensorflow::Scope device_scope =
              session->root()->WithDevice(xrt_device);
          const XrtSession::CachedNode& cached_node =
              GetAllocateNode(session, device_scope, device, tensors[i].shape);
          session_work->feed_inputs.insert({cached_node.holders[0], tensor});
          session_work->outputs_handles.push_back(cached_node.outputs[0]);
          session_work->index_mapping.push_back(i);

          total_size += tdata.size();
        }
      };
      env::ScheduleClosure(mwait.Completer(std::move(converter)));
    }
    mwait.Wait();
  }
  OutboundDataMetric()->AddSample(total_size);

  mwait.Reset(session_work_map.size());
  std::vector<DataPtr> results(tensors.size());
  for (auto& session_session_work : session_work_map) {
    XrtSession* session = session_session_work.first;
    SessionWork* session_work = &session_session_work.second;
    auto runner = [&, session, session_work]() {
      std::vector<tensorflow::Tensor> outputs;
      XLA_CHECK_OK(session->session()->Run(
          session_work->feed_inputs, session_work->outputs_handles, &outputs));
      XLA_CHECK_EQ(outputs.size(), session_work->outputs_handles.size());

      for (size_t i = 0; i < outputs.size(); ++i) {
        size_t li = session_work->index_mapping[i];
        results[li] = std::make_shared<XrtData>(device_ptr, tensors[li].shape,
                                                outputs[i].scalar<int64_t>()());
      }
      CreateDataHandlesCounter()->AddValue(outputs.size());
    };
    env::ScheduleIoClosure(mwait.Completer(std::move(runner)));
  }
  mwait.Wait();
  return results;
}

std::vector<Literal> XrtComputationClient::TransferFromServerImpl(
    absl::Span<const DataPtr> handles) {
  metrics::TimedSection timed(TransferFromServerMetric());

  int64_t max_partition_size = GetMaxTensorsPartitionSize();
  std::list<XrtSessionCache::SessionMap> session_maps;
  int64_t current_size = 0;
  session_maps.emplace_back();
  std::map<XrtSession*, SessionWork> session_work_map;
  for (size_t i = 0; i < handles.size(); ++i) {
    const XrtData& xrt_data = dynamic_cast<const XrtData&>(*handles[i]);

    int64_t shape_size = ShapeUtil::ByteSizeOfElements(xrt_data.shape());
    if (current_size + shape_size >= max_partition_size) {
      session_maps.emplace_back();
      current_size = 0;
    }
    current_size += shape_size;

    XrtSession* session = GetSessionForDevice(
        session_cache_.get(), xrt_data.device()->name(), &session_maps.back());
    SessionWork* session_work = &session_work_map[session];
    tensorflow::Scope device_scope = session->root()->WithDevice(
        SwiftDeviceToXrtDevice(xrt_data.device()->name()));
    const XrtSession::CachedNode& cached_node =
        GetReadNode(session, device_scope, xrt_data.device()->name());
    session_work->feed_inputs.insert(
        {cached_node.holders[0], xrt_data.get_handle()});
    session_work->outputs_handles.push_back(cached_node.outputs[0]);
    session_work->index_mapping.push_back(i);
  }

  int64_t total_size = 0;
  std::vector<Literal> results(handles.size());
  for (auto& session_work : session_work_map) {
    std::vector<tensorflow::Tensor> outputs;
    XLA_CHECK_OK(session_work.first->session()->Run(
        session_work.second.feed_inputs, session_work.second.outputs_handles,
        &outputs));
    XLA_CHECK_EQ(outputs.size(), session_work.second.outputs_handles.size());

    for (size_t i = 0; i < outputs.size(); ++i) {
      size_t li = session_work.second.index_mapping[i];
      LiteralProto response;
      XLA_CHECK(
          response.ParseFromString(outputs[i].scalar<tensorflow::tstring>()()));
      results[li] = std::move(Literal::CreateFromProto(response).ValueOrDie());
      total_size += results[li].size_bytes();
    }
  }
  InboundDataMetric()->AddSample(total_size);
  return results;
}

std::vector<ComputationClient::ComputationPtr> XrtComputationClient::Compile(
    const std::string& device, const std::vector<std::string>& devices,
    std::vector<CompileInstance> instances) {
  metrics::TimedSection timed(CompileMetric());

  std::mutex lock;
  util::MultiWait mwait(instances.size());
  std::vector<ProgramShape> program_shapes(instances.size());
  std::vector<ComputationPtr> results(instances.size());
  std::vector<CompilationCacheKey> cache_keys(instances.size());
  XrtSessionCache::SessionMap session_map;
  std::map<XrtSession*, SessionWork> session_work_map;
  for (size_t i = 0; i < instances.size(); ++i) {
    auto builder = [&, this, i]() {
      const CompileInstance& instance = instances[i];
      std::unique_ptr<xrt::XLAComputation> xrt_computation =
          CreateXrtComputation(instance.computation, devices,
                               instance.output_shape);
      CompilationCacheKey cache_key(GetResourceDomain(device),
                                    xrt_computation->SerializeAsString());
      auto computation_ptr = compilation_cache_.Get(cache_key);
      if (computation_ptr == nullptr) {
        cache_keys[i] = std::move(cache_key);
        program_shapes[i] =
            ProgramShape(xrt_computation->config().program_shape());

        const std::string& xrt_device = SwiftDeviceToXrtDevice(device);
        {
          std::lock_guard<std::mutex> slock(lock);
          XrtSession* session = GetSessionForXrtDevice(
              session_cache_.get(), xrt_device, &session_map);
          SessionWork* session_work = &session_work_map[session];
          tensorflow::Scope device_scope =
              session->root()->WithDevice(xrt_device);
          const XrtSession::CachedNode& cached_node =
              GetCompileNode(session, device_scope, device);
          session_work->feed_inputs.insert(
              {cached_node.holders[0], cache_keys[i].serialized_computation});
          session_work->outputs_handles.push_back(cached_node.outputs[0]);
          session_work->index_mapping.push_back(i);
        }
      } else {
        results[i] = computation_ptr;
      }
    };
    env::ScheduleClosure(mwait.Completer(std::move(builder)));
  }
  mwait.Wait();
  mwait.Reset(session_work_map.size());

  for (auto& session_and_work : session_work_map) {
    XrtSession* session = session_and_work.first;
    const SessionWork& session_work = session_and_work.second;

    auto session_runner = [&, this, session]() {
      std::vector<tensorflow::Tensor> outputs;
      CheckCompileStatus(
          session->session()->Run(session_work.feed_inputs,
                                  session_work.outputs_handles, &outputs),
          instances, session_work);
      XLA_CHECK_EQ(outputs.size(), session_work.outputs_handles.size());

      double compile_time = timed.Elapsed();
      size_t output_index = 0;
      for (auto li : session_work.index_mapping) {
        CompileInstance* instance = &instances[li];
        MaybeSaveLongCompileHlo(compile_time, instance->computation);
        results[li] = std::make_shared<XrtComputation>(
            this, std::move(instance->computation), program_shapes[li], devices,
            outputs[output_index].scalar<int64_t>()(), device);
        ++output_index;

        compilation_cache_.Add(std::move(cache_keys[li]), results[li]);
        CreateCompileHandlesCounter()->AddValue(1);
      }
    };
    env::ScheduleIoClosure(mwait.Completer(std::move(session_runner)));
  }
  mwait.Wait();
  return results;
}

void XrtComputationClient::CheckCompileStatus(
    const Status& status, const std::vector<CompileInstance>& instances,
    const SessionWork& session_work) {
  if (!status.ok()) {
    std::vector<const XlaComputation*> computations;
    std::vector<const Shape*> output_shapes;
    for (auto li : session_work.index_mapping) {
      computations.push_back(&instances[li].computation);
      output_shapes.push_back(instances[li].output_shape);
    }
    util::ReportComputationError(status, computations, output_shapes);
  }
}

std::vector<ComputationClient::DataPtr>
XrtComputationClient::ExecuteComputation(
    const Computation& computation, absl::Span<const DataPtr> arguments,
    const std::string& device, const ExecuteComputationOptions& options) {
  metrics::TimedSection timed(ExecuteMetric());

  XrtSessionCache::SessionMap session_map;
  std::string effective_device = GetEffectiveDevice(device);
  tensorflow::ClientSession::FeedType feed_inputs;
  std::vector<tensorflow::Output> exec_ops = CreateExecuteOps(
      &session_map, dynamic_cast<const XrtComputation&>(computation),
      BuildParallelArguments(arguments), options.explode_tuple,
      {effective_device}, &feed_inputs);

  XrtSession* session =
      GetSessionForDevice(session_cache_.get(), effective_device, &session_map);
  std::vector<tensorflow::Tensor> outputs;
  util::CheckComputationStatus(
      session->session()->Run(feed_inputs, {exec_ops.front()}, &outputs),
      {&computation.computation()}, {&computation.program_shape().result()});
  XLA_CHECK_EQ(outputs.size(), 1);

  return GetComputationResults(outputs[0], computation.program_shape().result(),
                               effective_device);
}

std::vector<std::vector<ComputationClient::DataPtr>>
XrtComputationClient::ExecuteReplicated(
    const Computation& computation,
    const std::vector<std::vector<DataPtr>>& arguments,
    absl::Span<const std::string> devices,
    const ExecuteReplicatedOptions& options) {
  metrics::TimedSection timed(ExecuteReplicatedMetric());

  XrtSessionCache::SessionMap session_map;
  tensorflow::ClientSession::FeedType feed_inputs;
  std::vector<tensorflow::Output> exec_ops = CreateExecuteOps(
      &session_map, dynamic_cast<const XrtComputation&>(computation), arguments,
      options.explode_tuple, devices, &feed_inputs);
  std::vector<const Computation*> computations(devices.size());
  std::fill(computations.begin(), computations.end(), &computation);

  return RunComputations(session_map, exec_ops, computations, devices,
                         feed_inputs);
}

std::vector<std::vector<ComputationClient::DataPtr>>
XrtComputationClient::RunComputations(
    const XrtSessionCache::SessionMap& session_map,
    const std::vector<tensorflow::Output>& exec_ops,
    absl::Span<const Computation* const> computations,
    absl::Span<const std::string> devices,
    const tensorflow::ClientSession::FeedType& feed_inputs) {
  // In the S4TF/XRT interface we keep a map (options_.workers_map) from a
  // worker+taskno, to the GRPC server which is the entry point for that worker.
  // Since XRT could re-distribute ops internally, if we have N hosts
  // (worker+taskno), we could have all the workers pointing to a single GRPC
  // entry point, or we could have each worker pointing directly to the target
  // host.
  // The advantage of the latter approach, is that we do not bottleneck
  // (especially when feeding inputs) the single GRPC entry point.
  // Using the N:1 approach, the session_replicas below will contain a single
  // session, and all the replica executions will go through it (and distributed
  // by XRT on the service side).
  // Chosing the 1:1 approach (one session per worker), we will have N sessions
  // within the session_replicas map, which we will be executing independently.
  std::map<XrtSession*, std::vector<size_t>> session_replicas;
  for (size_t i = 0; i < devices.size(); ++i) {
    auto worker_hostport = GetWorkerForDevice(GetEffectiveDevice(devices[i]));
    XrtSession* session = session_map.at(worker_hostport.second).get();
    session_replicas[session].push_back(i);
  }
  XLA_CHECK_EQ(computations.size(), devices.size());

  util::MultiWait mwait(session_replicas.size());
  std::vector<std::vector<DataPtr>> results(devices.size());
  for (auto& sess_replica : session_replicas) {
    XrtSession* session = sess_replica.first;
    const std::vector<size_t>& replicas = sess_replica.second;

    auto session_runner = [&, this, session]() {
      std::vector<tensorflow::Output> exec_nodes;
      std::vector<const XlaComputation*> xla_computations;
      std::vector<const Shape*> output_shapes;
      for (auto replica : replicas) {
        exec_nodes.push_back(exec_ops[replica]);
        xla_computations.push_back(&computations[replica]->computation());
        output_shapes.push_back(
            &computations[replica]->program_shape().result());
      }
      std::vector<tensorflow::Tensor> outputs;
      util::CheckComputationStatus(
          session->session()->Run(feed_inputs, exec_nodes, &outputs),
          xla_computations, output_shapes);
      XLA_CHECK_EQ(outputs.size(), exec_nodes.size());

      for (size_t i = 0; i < outputs.size(); ++i) {
        auto replica = replicas[i];
        results[replica] = GetComputationResults(
            outputs[i], computations[replica]->program_shape().result(),
            GetEffectiveDevice(devices[replica]));
      }
    };
    env::ScheduleIoClosure(mwait.Completer(std::move(session_runner)));
  }
  mwait.Wait();
  return results;
}

std::vector<std::vector<ComputationClient::DataPtr>>
XrtComputationClient::ExecuteParallel(
    absl::Span<const Computation* const> computations,
    const std::vector<std::vector<DataPtr>>& arguments,
    absl::Span<const std::string> devices,
    const ExecuteParallelOptions& options) {
  metrics::TimedSection timed(ExecuteParallelMetric());

  XrtSessionCache::SessionMap session_map;
  tensorflow::ClientSession::FeedType feed_inputs;
  std::vector<tensorflow::Output> exec_ops =
      CreateExecuteOps(&session_map, computations, arguments,
                       options.explode_tuple, devices, &feed_inputs);
  return RunComputations(session_map, exec_ops, computations, devices,
                         feed_inputs);
}

std::vector<ComputationClient::DataPtr> XrtComputationClient::ExecuteChained(
    absl::Span<const ExecuteChainedOp> ops, const std::string& device) {
  static int64_t split_mode = sys_util::GetEnvInt("XRT_SPLIT_CHAINED_EXEC", 0);
  return split_mode ? ExecuteChainedSplit(ops, device)
                    : ExecuteChainedXrt(ops, device);
}

std::vector<ComputationClient::DataPtr> XrtComputationClient::ExecuteChainedXrt(
    absl::Span<const ExecuteChainedOp> ops, const std::string& device) {
  metrics::TimedSection timed(ExecuteChainedMetric());

  XrtSessionCache::SessionMap session_map;
  std::string effective_device = GetEffectiveDevice(device);
  const std::string& xrt_device = SwiftDeviceToXrtDevice(effective_device);
  tensorflow::ClientSession::FeedType feed_inputs;
  XrtSession* session =
      GetSessionForXrtDevice(session_cache_.get(), xrt_device, &session_map);
  tensorflow::Scope device_scope = session->root()->WithDevice(xrt_device);

  xrt::XRTChainedExecuteConfig config;
  config.set_core_index_in_replica(0);
  config.set_rng_seed(rng_seed_);

  xrt::XRTChainedExecutePlan plan;
  std::vector<xla::Shape> result_shapes;
  for (size_t i = 0; i < ops.size(); ++i) {
    const ExecuteChainedOp& op = ops[i];
    xrt::XRTChainedExecuteOp* plan_op = plan.add_ops();
    const xla::Shape* op_shape = nullptr;
    if (op.device_data != nullptr) {
      const XrtData& xrt_data = dynamic_cast<const XrtData&>(*op.device_data);
      op_shape = &xrt_data.shape();
      plan_op->set_data_handle(xrt_data.get_handle());
    } else {
      const XrtComputation& xrt_computation =
          dynamic_cast<const XrtComputation&>(*op.computation);
      op_shape = &xrt_computation.program_shape().result();
      plan_op->set_computation_handle(xrt_computation.get_handle());
      for (auto& input : op.inputs) {
        XLA_CHECK_LT(input.op_index, i);

        xrt::XRTChainedExecuteOp::Input* plan_input = plan_op->add_inputs();
        plan_input->set_op_index(input.op_index);
        if (input.output_index) {
          plan_input->set_output_index(*input.output_index + 1);
        }
      }
    }
    for (auto& output : op.outputs) {
      XLA_CHECK(op_shape != nullptr);

      xrt::XRTChainedExecuteOp::Output* plan_output = plan_op->add_outputs();
      plan_output->set_result_index(output.result_index);
      if (output.result_index >= result_shapes.size()) {
        result_shapes.resize(output.result_index + 1);
      }
      if (output.output_index) {
        plan_output->set_output_index(*output.output_index + 1);
        result_shapes[output.result_index] =
            ShapeUtil::GetTupleElementShape(*op_shape, *output.output_index);
      } else {
        result_shapes[output.result_index] = *op_shape;
      }
    }
  }

  const XrtSession::CachedNode& cached_node =
      GetExecuteChainedNode(session, device_scope, effective_device);
  feed_inputs.insert({cached_node.holders[0], plan.SerializeAsString()});
  feed_inputs.insert({cached_node.holders[1], config.SerializeAsString()});

  std::vector<tensorflow::Tensor> outputs;
  util::CheckComputationStatus(
      session->session()->Run(feed_inputs, {cached_node.outputs[0]}, &outputs),
      {}, {});
  XLA_CHECK_EQ(outputs.size(), 1);

  std::vector<DataPtr> results;
  auto handles_vec = outputs[0].vec<int64_t>();
  for (int64_t i = 0; i < handles_vec.size(); ++i) {
    results.push_back(std::make_shared<XrtData>(
        dynamic_cast<XrtDevice*>(GetDevice(effective_device)),
        std::move(result_shapes.at(i)), handles_vec(i)));
  }
  CreateDataHandlesCounter()->AddValue(results.size());
  return results;
}

std::vector<ComputationClient::DataPtr>
XrtComputationClient::ExecuteChainedSplit(
    absl::Span<const ExecuteChainedOp> ops, const std::string& device) {
  metrics::TimedSection timed(ExecuteChainedMetric());

  std::vector<int64_t> uses(ops.size(), 0);
  for (auto& op : ops) {
    for (auto& input : op.inputs) {
      uses[input.op_index] += 1;
    }
  }
  XrtSessionCache::SessionMap session_map;
  std::string effective_device = GetEffectiveDevice(device);
  const std::string& xrt_device = SwiftDeviceToXrtDevice(effective_device);
  XrtSession* session =
      GetSessionForXrtDevice(session_cache_.get(), xrt_device, &session_map);
  tensorflow::Scope device_scope = session->root()->WithDevice(xrt_device);
  std::vector<std::vector<DataPtr>> ops_outputs(ops.size());
  std::vector<DataPtr> results;
  for (size_t i = 0; i < ops.size(); ++i) {
    const ExecuteChainedOp& op = ops[i];
    if (op.device_data != nullptr) {
      ops_outputs[i].push_back(op.device_data);
    } else {
      tensorflow::ClientSession::FeedType feed_inputs;
      std::vector<DataPtr> arguments;
      arguments.reserve(op.inputs.size());
      for (auto& input : op.inputs) {
        XLA_CHECK_LT(input.op_index, i);
        XLA_CHECK_LT(input.output_index.value_or(0),
                     ops_outputs[input.op_index].size());
        arguments.push_back(
            ops_outputs[input.op_index][input.output_index.value_or(0)]);
      }

      std::vector<tensorflow::Output> exec_ops = CreateExecuteOps(
          &session_map, dynamic_cast<const XrtComputation&>(*op.computation),
          BuildParallelArguments(arguments), /*explode_tuple=*/true,
          {effective_device}, &feed_inputs);

      std::vector<tensorflow::Tensor> outputs;
      util::CheckComputationStatus(
          session->session()->Run(feed_inputs, {exec_ops.front()}, &outputs),
          {&op.computation->computation()},
          {&op.computation->program_shape().result()});
      XLA_CHECK_EQ(outputs.size(), 1);
      ops_outputs[i] = GetComputationResults(
          outputs[0], op.computation->program_shape().result(),
          effective_device);
    }

    for (auto& output : op.outputs) {
      if (output.result_index >= results.size()) {
        results.resize(output.result_index + 1);
      }
      XLA_CHECK_LT(output.output_index.value_or(0), ops_outputs[i].size());
      results[output.result_index] =
          ops_outputs[i][output.output_index.value_or(0)];
    }
    // Drop references to any intermediate result which is not used anymore.
    for (auto& input : op.inputs) {
      uses[input.op_index] -= 1;
      if (uses[input.op_index] == 0) {
        ops_outputs[input.op_index].clear();
      }
    }
    // We can reset the TF op cache here so that we don't keep allocating new
    // TF op nodes on the session graph.
    session->Reset();
  }
  return results;
}

std::vector<std::vector<ComputationClient::DataPtr>>
XrtComputationClient::DeconstructTuple(absl::Span<const DataPtr> tuples) {
  metrics::TimedSection timed(DeconstructTupleMetric());

  XrtSessionCache::SessionMap session_map;
  std::map<XrtSession*, SessionWork> session_work_map;
  std::vector<int64_t> tuple_elements_count(tuples.size());
  for (size_t i = 0; i < tuples.size(); ++i) {
    const XrtData& xrt_data = dynamic_cast<const XrtData&>(*tuples[i]);
    XrtSession* session = GetSessionForDevice(
        session_cache_.get(), xrt_data.device()->name(), &session_map);
    SessionWork* session_work = &session_work_map[session];
    session_work->index_mapping.push_back(i);

    tensorflow::Scope device_scope = session->root()->WithDevice(
        SwiftDeviceToXrtDevice(xrt_data.device()->name()));
    int64_t count = ShapeUtil::TupleElementCount(xrt_data.shape());
    tuple_elements_count[i] = count;
    for (int64_t j = 0; j < count; ++j) {
      const XrtSession::CachedNode& cached_node =
          GetSubTupleNode(session, device_scope, xrt_data.device()->name());
      session_work->feed_inputs.insert(
          {cached_node.holders[0], xrt_data.get_handle()});
      tensorflow::Tensor index_tensor(tensorflow::DT_INT32,
                                      tensorflow::TensorShape({1}));
      index_tensor.flat<tensorflow::int32>()(0) = j;
      session_work->feed_inputs.insert({cached_node.holders[1], index_tensor});
      session_work->outputs_handles.push_back(cached_node.outputs[0]);
    }
  }

  std::vector<std::vector<DataPtr>> results(tuples.size());
  for (auto& session_work : session_work_map) {
    std::vector<tensorflow::Tensor> outputs;
    XLA_CHECK_OK(session_work.first->session()->Run(
        session_work.second.feed_inputs, session_work.second.outputs_handles,
        &outputs));
    XLA_CHECK_EQ(outputs.size(), session_work.second.outputs_handles.size());

    size_t output_index = 0;
    for (auto li : session_work.second.index_mapping) {
      const XrtData& xrt_data = dynamic_cast<const XrtData&>(*tuples[li]);
      std::vector<DataPtr> tuple_results;
      for (size_t i = 0; i < tuple_elements_count[li]; ++i, ++output_index) {
        tuple_results.push_back(std::make_shared<XrtData>(
            dynamic_cast<XrtDevice*>(xrt_data.device()),
            ShapeUtil::GetTupleElementShape(xrt_data.shape(), i),
            outputs[output_index].scalar<int64_t>()()));
      }
      results[li] = std::move(tuple_results);
      CreateDataHandlesCounter()->AddValue(tuple_elements_count[li]);
    }
  }
  return results;
}

XrtSession* XrtComputationClient::GetSessionForTarget(
    XrtSessionCache* cache, const std::string& target,
    XrtSessionCache::SessionMap* session_map) {
  return cache->GetSession(target, session_map);
}

XrtSession* XrtComputationClient::GetSessionForXrtDevice(
    XrtSessionCache* cache, const std::string& xrt_device,
    XrtSessionCache::SessionMap* session_map) {
  auto worker_hostport = GetWorkerForXrtDevice(xrt_device);
  return GetSessionForTarget(cache, worker_hostport.second, session_map);
}

XrtSession* XrtComputationClient::GetSessionForDevice(
    XrtSessionCache* cache, const std::string& device,
    XrtSessionCache::SessionMap* session_map) {
  return GetSessionForXrtDevice(cache, SwiftDeviceToXrtDevice(device),
                                session_map);
}

std::string XrtComputationClient::GetEffectiveDevice(
    const std::string& device) const {
  if (device.empty()) {
    return options_.default_device;
  }
  if (device[0] == ':') {
    // Allow devices with ordinal only specification, to expand from the default
    // device type.
    auto pos = options_.default_device.find(':');
    XLA_CHECK_NE(pos, std::string::npos) << options_.default_device;
    return options_.default_device.substr(0, pos) + device;
  }
  return device;
}

const std::string& XrtComputationClient::SwiftDeviceToXrtDevice(
    const std::string& device) const {
  auto device_target =
      options_.global_device_map.find(GetEffectiveDevice(device));
  XLA_CHECK(device_target != options_.global_device_map.end())
      << "Unable to find device: " << device;
  return device_target->second;
}

std::unique_ptr<xrt::XLAComputation> XrtComputationClient::CreateXrtComputation(
    const XlaComputation& computation, absl::Span<const std::string> devices,
    const Shape* output_shape) const {
  std::unique_ptr<xrt::XLAComputation> xrt_computation(
      new xrt::XLAComputation());
  auto config = xrt_computation->mutable_config();
  config->set_num_cores_per_replica(1);
  if (devices.size() > 1) {
    auto device_assignment = config->mutable_device_assignment();
    auto computation_device = device_assignment->add_computation_devices();
    for (int64_t i = 0; i < devices.size(); ++i) {
      DeviceId device(devices[i]);
      auto replica_device = computation_device->add_replica_devices();
      if (device.kind == "TPU") {
        const std::string& xrt_device = SwiftDeviceToXrtDevice(devices[i]);
        const auto& core_coords = GetDeviceMeshCoords(xrt_device);
        for (auto coord : core_coords) {
          replica_device->add_value(coord);
        }
      } else if (device.kind == "GPU") {
        // For GPU use X,Y,Z=0 and CORE=GPU_ORDINAL (where GPU_ORDINAL is the
        // global ordinal value).
        replica_device->add_value(0);
        replica_device->add_value(0);
        replica_device->add_value(0);
        replica_device->add_value(device.ordinal);
      } else {
        XLA_ERROR() << "Unsupported replication device type: " << device.kind;
      }
    }
    config->set_num_replicas(devices.size());
  }
  *config->mutable_program_shape() =
      computation.GetProgramShape().ValueOrDie().ToProto();
  if (output_shape != nullptr) {
    *config->mutable_program_shape()->mutable_result() =
        output_shape->ToProto();
  }
  *xrt_computation->mutable_hlo_snapshot() =
      std::move(*computation.Snapshot().ConsumeValueOrDie());
  return xrt_computation;
}

tensorflow::Tensor XrtComputationClient::GetArgumentsInputs(
    absl::Span<const DataPtr> arguments, const std::string& device) {
  tensorflow::Tensor inputs_tensor(
      tensorflow::DT_INT64,
      tensorflow::TensorShape({static_cast<int64_t>(arguments.size())}));
  for (size_t i = 0; i < arguments.size(); ++i) {
    const XrtData& xrt_data = dynamic_cast<const XrtData&>(*arguments[i]);
    XLA_CHECK_EQ(device, xrt_data.device()->name());
    inputs_tensor.flat<tensorflow::int64>()(i) = xrt_data.get_handle();
  }
  return inputs_tensor;
}

std::vector<tensorflow::Output> XrtComputationClient::CreateExecuteOps(
    XrtSessionCache::SessionMap* session_map,
    absl::Span<const Computation* const> computations,
    const std::vector<std::vector<DataPtr>>& arguments, bool explode_tuple,
    absl::Span<const std::string> devices,
    tensorflow::ClientSession::FeedType* feed_inputs) {
  std::vector<tensorflow::Output> exec_ops;
  for (size_t i = 0; i < computations.size(); ++i) {
    const XrtComputation* xrt_computation =
        dynamic_cast<const XrtComputation*>(computations[i]);
    auto inputs = GetArgumentsInputs(arguments[i], devices[i]);
    const std::string& xrt_device = SwiftDeviceToXrtDevice(devices[i]);
    XrtSession* session =
        GetSessionForXrtDevice(session_cache_.get(), xrt_device, session_map);
    tensorflow::Scope device_scope = session->root()->WithDevice(xrt_device);
    const XrtSession::CachedNode& cached_node =
        GetExecuteNode(session, device_scope, devices[i]);
    feed_inputs->insert(
        {cached_node.holders[0], xrt_computation->get_handle()});

    xrt::XRTExecutionConfig exec_config;
    exec_config.set_core_index_in_replica(0);
    exec_config.set_release_input_handles(false);
    exec_config.set_release_compilation_handle(false);
    exec_config.set_return_exploded_tuple(explode_tuple);
    exec_config.set_rng_seed(rng_seed_);
    feed_inputs->insert(
        {cached_node.holders[1], exec_config.SerializeAsString()});
    feed_inputs->insert({cached_node.holders[2], inputs});

    exec_ops.push_back(cached_node.outputs[0]);
  }
  return exec_ops;
}

std::vector<tensorflow::Output> XrtComputationClient::CreateExecuteOps(
    XrtSessionCache::SessionMap* session_map, const XrtComputation& computation,
    const std::vector<std::vector<DataPtr>>& arguments, bool explode_tuple,
    absl::Span<const std::string> devices,
    tensorflow::ClientSession::FeedType* feed_inputs) {
  std::vector<tensorflow::Output> exec_ops;
  for (size_t i = 0; i < arguments.size(); ++i) {
    auto inputs = GetArgumentsInputs(arguments[i], devices[i]);
    const std::string& xrt_device = SwiftDeviceToXrtDevice(devices[i]);
    XrtSession* session =
        GetSessionForXrtDevice(session_cache_.get(), xrt_device, session_map);
    tensorflow::Scope device_scope = session->root()->WithDevice(xrt_device);
    const XrtSession::CachedNode& cached_node =
        GetExecuteNode(session, device_scope, devices[i]);
    feed_inputs->insert({cached_node.holders[0], computation.get_handle()});

    xrt::XRTExecutionConfig exec_config;
    exec_config.set_core_index_in_replica(0);
    exec_config.set_release_input_handles(false);
    exec_config.set_release_compilation_handle(false);
    exec_config.set_return_exploded_tuple(explode_tuple);
    exec_config.set_rng_seed(rng_seed_);
    feed_inputs->insert(
        {cached_node.holders[1], exec_config.SerializeAsString()});
    feed_inputs->insert({cached_node.holders[2], inputs});

    exec_ops.push_back(cached_node.outputs[0]);
  }
  return exec_ops;
}

void XrtComputationClient::ReleaseHandles(
    std::vector<DeviceHandle>* handles,
    const std::function<const XrtSession::CachedNode&(
        XrtSession*, const tensorflow::Scope&, const std::string&)>&
        op_generator,
    metrics::Metric* timed_metric, metrics::Counter* destroy_counter) {
  std::vector<DeviceHandle> released_handles;
  {
    std::lock_guard<std::mutex> lock(lock_);
    released_handles.swap(*handles);
  }
  if (!released_handles.empty()) {
    metrics::TimedSection timed(timed_metric);

    XrtSessionCache::SessionMap session_map;
    std::map<XrtSession*, std::vector<DeviceHandle>> session_handles_map;
    for (auto& handle : released_handles) {
      XrtSession* session = GetSessionForDevice(session_cache_.get(),
                                                handle.device, &session_map);
      session_handles_map[session].push_back(handle);
    }
    for (const auto& session_and_handles : session_handles_map) {
      XrtSession* session = session_and_handles.first;
      const std::vector<DeviceHandle>& session_handles =
          session_and_handles.second;
      tensorflow::Tensor handles_tensor(
          tensorflow::DT_INT64, tensorflow::TensorShape({static_cast<int64_t>(
                                    session_handles.size())}));
      auto flat_handles_tensor = handles_tensor.flat<tensorflow::int64>();
      for (size_t i = 0; i < session_handles.size(); ++i) {
        flat_handles_tensor(i) = session_handles[i].handle;
      }
      tensorflow::Scope device_scope = session->root()->WithDevice(
          SwiftDeviceToXrtDevice(session_handles.front().device));
      const XrtSession::CachedNode& cached_node =
          op_generator(session, device_scope, session_handles.front().device);
      tensorflow::ClientSession::FeedType feed_inputs;
      feed_inputs.insert({cached_node.holders[0], handles_tensor});

      std::vector<tensorflow::Tensor> outputs;
      XLA_CHECK_OK(session->session()->Run(
          feed_inputs, {}, {cached_node.operations[0]}, &outputs));
    }
    destroy_counter->AddValue(released_handles.size());
  }
}

void XrtComputationClient::StartHandleReleaser() {
  static const size_t kMinReleaserThreads = 8;
  int64_t num_threads = sys_util::GetEnvInt(
      "XLA_HANDLE_RELEASE_THREADS",
      std::max<size_t>(options_.devices.size(), kMinReleaserThreads));
  triggered_task_ = absl::make_unique<util::TriggeredTask>(
      [this]() { HandleReleaser(); }, num_threads);
}

void XrtComputationClient::HandleReleaser() {
  auto data_op_generator =
      [this](XrtSession* session, const tensorflow::Scope& scope,
             const std::string& device) -> const XrtSession::CachedNode& {
    return GetReleaseAllocationHandleNode(session, scope, device);
  };
  ReleaseHandles(&released_data_handles_, data_op_generator,
                 ReleaseDataHandlesTimeMetric(), DestroyDataHandlesCounter());

  auto compile_op_generator =
      [this](XrtSession* session, const tensorflow::Scope& scope,
             const std::string& device) -> const XrtSession::CachedNode& {
    return GetReleaseCompileHandleNode(session, scope, device);
  };
  ReleaseHandles(&released_compile_handles_, compile_op_generator,
                 ReleaseCompileHandlesTimeMetric(),
                 DestroyCompileHandlesCounter());
}

void XrtComputationClient::ReleaseHandle(int64_t handle,
                                         const std::string& device,
                                         std::vector<DeviceHandle>* handles) {
  {
    std::lock_guard<std::mutex> lock(lock_);
    handles->push_back({device, handle});
  }
  triggered_task_->Activate();
}

void XrtComputationClient::ReleaseXrtData(const std::string& device,
                                          int64_t handle) {
  ReleaseHandle(handle, device, &released_data_handles_);
  ReleaseDataHandlesCounter()->AddValue(1);
}

void XrtComputationClient::ReleaseXrtComputation(
    const std::string& compilation_device, int64_t handle) {
  ReleaseHandle(handle, compilation_device, &released_compile_handles_);
  ReleaseCompileHandlesCounter()->AddValue(1);
}

std::pair<XrtComputationClient::Worker, std::string>
XrtComputationClient::GetWorkerForXrtDevice(
    const std::string& xrt_device) const {
  tensorflow::DeviceNameUtils::ParsedName parsed_device =
      ParseFullXrtDevice(xrt_device);
  auto worker_hostport =
      options_.workers_map.find(Worker(parsed_device.job, parsed_device.task));
  XLA_CHECK(worker_hostport != options_.workers_map.end()) << xrt_device;
  return std::pair<Worker, std::string>(worker_hostport->first,
                                        worker_hostport->second);
}

std::pair<XrtComputationClient::Worker, std::string>
XrtComputationClient::GetWorkerForDevice(const std::string& device) const {
  return GetWorkerForXrtDevice(SwiftDeviceToXrtDevice(device));
}

const std::vector<int>& XrtComputationClient::GetDeviceMeshCoords(
    const std::string& xrt_device) const {
  auto it = device_mesh_coords_.find(xrt_device);
  if (it == device_mesh_coords_.end()) {
    TF_LOG(FATAL) << "Missing mesh coordinates for device: " << xrt_device;
  }
  return it->second;
}

tensorflow::tpu::TopologyProto XrtComputationClient::InitializeAndFetchTopology(
    const std::string& job, int task_no, const std::string& worker_host_port,
    const tensorflow::ConfigProto& config) {
  tensorflow::SessionOptions session_options;
  session_options.env = tensorflow::Env::Default();
  session_options.target = worker_host_port;
  session_options.config = config;

  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ClientSession session(root, session_options);
  std::string system_device = absl::StrCat(
      "/job:", job, "/replica:0/task:", task_no, "/device:TPU_SYSTEM:0");
  tensorflow::Scope tpu_system_scope = root.WithDevice(system_device);
  const auto unique_name =
      tpu_system_scope.GetUniqueNameForOp("ConfigureDistributedTPU");
  tensorflow::NodeBuilder builder =
      tensorflow::NodeBuilder(unique_name, "ConfigureDistributedTPU")
          .Attr("embedding_config", "")
          .Attr("tpu_embedding_config", "")
          .Attr("is_global_init", false);
  // TODO(asuhan): Remove this once the new TF build can be relied upon, on the
  // Cloud TPU side.
  const tensorflow::ClusterDef cluster_def = config.cluster_def();
  if (cluster_def.job_size() > 1 ||
      (cluster_def.job_size() == 1 && cluster_def.job()[0].tasks_size() > 1)) {
    builder.Attr("enable_whole_mesh_compilations", true);
  }

  tpu_system_scope.UpdateBuilder(&builder);

  tensorflow::Node* result;
  root.UpdateStatus(builder.Finalize(tpu_system_scope.graph(), &result));
  XLA_CHECK_OK(tpu_system_scope.status());
  root.UpdateStatus(tpu_system_scope.DoShapeInference(result));

  std::vector<tensorflow::Tensor> outputs;
  XLA_CHECK_OK(root.status());
  XLA_CHECK_OK(session.Run({tensorflow::Output(result, 0)}, &outputs));
  XLA_CHECK_EQ(outputs.size(), 1);

  return ParseProto<tensorflow::tpu::TopologyProto>(outputs[0]);
}

void XrtComputationClient::InitializeDevices(
    std::unique_ptr<tensorflow::tpu::TopologyProto> topology_proto) {
  if (topology_proto == nullptr) {
    std::set<Worker> tpu_workers;
    for (const auto& dev_target : options_.global_device_map) {
      tensorflow::DeviceNameUtils::ParsedName parsed_device =
          ParseFullXrtDevice(dev_target.second);
      if (parsed_device.type == "TPU") {
        tpu_workers.emplace(parsed_device.job, parsed_device.task);
      }
    }
    if (!tpu_workers.empty()) {
      const Worker& worker = *tpu_workers.begin();
      auto it = options_.workers_map.find(worker);
      XLA_CHECK(it != options_.workers_map.end());

      TF_VLOG(1) << "Configuring TPU for worker " << worker.name << ":"
                 << worker.task_no << " at " << it->second;
      tensorflow::tpu::TopologyProto worker_topology_proto =
          InitializeAndFetchTopology(worker.name, worker.task_no, it->second,
                                     session_cache_->GetConfig());
      if (topology_proto == nullptr) {
        topology_proto = absl::make_unique<tensorflow::tpu::TopologyProto>(
            std::move(worker_topology_proto));
      }
    }
    if (topology_proto != nullptr) {
      TF_VLOG(1) << "TPU topology: " << topology_proto->DebugString();
    }
  }
  for (const auto& dev_target : options_.global_device_map) {
    tensorflow::DeviceNameUtils::ParsedName parsed_device =
        ParseFullXrtDevice(dev_target.second);
    if (parsed_device.type != "TPU") {
      continue;
    }
    XLA_CHECK_LE(parsed_device.task, topology_proto->num_tasks());
    XLA_CHECK_LE(parsed_device.id, topology_proto->num_tpu_devices_per_task());
    // The topology proto 'device_coordinates' is a linear list of
    // [num_tasks][devices_per_task][mesh_shape_size] coordinates, where the
    // mesh coordinates are usually [x, y, z, c] ('x', 'y' and 'z' being the
    // spatial chip coordinated and 'c' the core number).
    int64_t base_index = parsed_device.task *
                           topology_proto->num_tpu_devices_per_task() *
                           topology_proto->mesh_shape_size() +
                       parsed_device.id * topology_proto->mesh_shape_size();
    std::vector<int> device_mesh_coords(topology_proto->mesh_shape_size());
    for (int i = 0; i < topology_proto->mesh_shape_size(); ++i) {
      device_mesh_coords[i] =
          topology_proto->device_coordinates(base_index + i);
    }
    device_mesh_coords_.insert(
        {dev_target.second, std::move(device_mesh_coords)});
  }

  // Create the mesh service only if we have more than one worker, or if
  // multi-processing is active.
  std::string mesh_service_address =
      sys_util::GetEnvString(env::kEnvMeshService, "");
  std::string mp_device = GetMultiProcessingDevice();
  if (!mesh_service_address.empty() && !mp_device.empty()) {
    DeviceId device(mp_device);
    if (device.ordinal == 0) {
      CreateMeshService(mesh_service_address, topology_proto.get());
    }
    SetupGpuRuntime();
  }
}

void XrtComputationClient::SetupGpuRuntime() {
  LOG(FATAL) << "Not implemented yet; need to upgrade XRT first";
}

void XrtComputationClient::CreateMeshService(
    const std::string& address,
    const tensorflow::tpu::TopologyProto* topology_proto) {
  struct Device {
    std::string local_name;
    std::string global_name;
  };

  service::grpc::Config config;
  if (topology_proto != nullptr) {
    *config.mutable_proto() = *topology_proto;
  }

  std::map<Worker, std::vector<Device>> workers_devices;
  for (const auto& dev_target : options_.global_device_map) {
    tensorflow::DeviceNameUtils::ParsedName parsed_device =
        ParseFullXrtDevice(dev_target.second);
    std::string local_name =
        absl::StrCat(parsed_device.type, ":", parsed_device.id);
    workers_devices[Worker(parsed_device.job, parsed_device.task)].push_back(
        {local_name, dev_target.first});
  }
  for (auto& worker_address : options_.workers_map) {
    service::grpc::Worker* worker = config.add_workers();
    worker->set_name(worker_address.first.name);
    worker->set_task_no(worker_address.first.task_no);
    worker->set_address(worker_address.second);
    for (auto& worker_device : workers_devices[worker_address.first]) {
      service::grpc::Device* device = worker->add_devices();
      device->set_local_name(worker_device.local_name);
      device->set_global_name(worker_device.global_name);
    }
  }
  config.set_mesh_size(sys_util::GetEnvInt(env::kEnvWorldSize, 1));

  TF_VLOG(1) << "Creating mesh service bound to " << address;
  mesh_service_ =
      absl::make_unique<service::MeshService>(address, std::move(config));
}

std::vector<ComputationClient::DataPtr>
XrtComputationClient::GetComputationResults(
    const tensorflow::Tensor& xrt_result, const Shape& result_shape,
    const std::string& device_name) {
  std::vector<DataPtr> results;
  auto* device = dynamic_cast<XrtDevice*>(GetDevice(device_name));
  if (xrt_result.dims() == 1) {
    auto handles_vec = xrt_result.vec<int64_t>();
    for (int64_t i = 0; i < handles_vec.size(); ++i) {
      results.push_back(std::make_shared<XrtData>(
          device, ShapeUtil::GetTupleElementShape(result_shape, i),
          handles_vec(i)));
    }
  } else {
    results.push_back(std::make_shared<XrtData>(device, result_shape,
                                                xrt_result.scalar<int64_t>()()));
  }
  CreateDataHandlesCounter()->AddValue(results.size());
  return results;
}

std::string XrtComputationClient::GetResourceDomain(
    const std::string& device) const {
  return GetWorkerForDevice(device).second;
}

std::string XrtComputationClient::GetDefaultDevice() const {
  return options_.default_device;
}

size_t XrtComputationClient::GetNumDevices() const {
  return options_.devices.size();
}

std::vector<std::string> XrtComputationClient::GetLocalDevices() const {
  return std::vector<std::string>(options_.devices.begin(),
                                  options_.devices.end());
}

void XrtComputationClient::SetRngSeed(size_t seed) { rng_seed_ = seed; }

std::map<std::string, Metric> XrtComputationClient::GetMetrics() const {
  static const std::map<std::string, std::string>* metric_remap =
      new std::map<std::string, std::string>{
          {"/tensorflow/xrt/ops/allocate", "XrtAllocate"},
          {"/tensorflow/xrt/ops/allocate_from_tensor", "XrtAllocateFromTensor"},
          {"/tensorflow/xrt/ops/sub_tuple", "XrtSubTuple"},
          {"/tensorflow/xrt/ops/make_tuple", "XrtMakeTuple"},
          {"/tensorflow/xrt/ops/compile", "XrtCompile"},
          {"/tensorflow/xrt/ops/release_compilation", "XrtReleaseCompilation"},
          {"/tensorflow/xrt/ops/execute", "XrtExecute"},
          {"/tensorflow/xrt/ops/execute_chained", "XrtExecuteChained"},
          {"/tensorflow/xrt/ops/read_literal", "XrtReadLiteral"},
          {"/tensorflow/xrt/ops/read_tensor", "XrtReadTensor"},
          {"/tensorflow/xrt/ops/write_literal", "XrtWriteLiteral"},
          {"/tensorflow/xrt/ops/release_allocation", "XrtReleaseAllocation"},
          {"/tensorflow/xrt/ops/release_all_allocations",
           "XrtReleaseAllAllocations"},
          {"/tensorflow/xrt/ops/compact_allocations", "XrtCompactAllocations"},
          {"/tensorflow/xrt/memory_manager/compaction", "XrtCompaction"},
          {"/tensorflow/xrt/memory_manager/try_free_memory",
           "XrtTryFreeMemory"},
          {"/tensorflow/xrt/executor/program_memory_evict", "XrtExecutorEvict"},
          {"/tensorflow/xrt/ds_executor/program_memory_evict",
           "XrtExecutorEvict"}};

  std::map<std::string, Metric> metrics_data;
  xrt::XRTMetricsCollect metrics;
  metrics.add_metrics_regex("/tensorflow/xrt/.*");

  for (auto& worker_target : options_.workers_map) {
    tensorflow::SessionOptions session_options;
    session_options.env = tensorflow::Env::Default();
    session_options.target = worker_target.second;
    session_options.config = session_cache_->GetConfig();

    tensorflow::Scope root = tensorflow::Scope::NewRootScope();
    tensorflow::ClientSession session(root, session_options);
    std::string cpu0_device = absl::StrCat(
        "/job:", worker_target.first.name,
        "/replica:0/task:", worker_target.first.task_no, "/device:CPU:0");
    tensorflow::Scope cpu_system_scope = root.WithDevice(cpu0_device);
    auto metrics_value =
        tensorflow::ops::Const(cpu_system_scope, metrics.SerializeAsString());
    tensorflow::Output result =
        tensorflow::ops::XRTMetricsCollect(cpu_system_scope, metrics_value);
    XLA_CHECK_OK(cpu_system_scope.status());

    std::vector<tensorflow::Tensor> outputs;
    XLA_CHECK_OK(session.Run({result}, &outputs));
    XLA_CHECK_EQ(outputs.size(), 1);

    xrt::MetricsReport report = ParseProto<xrt::MetricsReport>(outputs[0]);
    for (auto& xrt_metric : report.metrics()) {
      Metric metric;
      if (xrt_metric.values_oneof_case() ==
          xrt::MetricValues::kPercentilesValue) {
        const xrt::Percentiles& xrt_percentile = xrt_metric.percentiles_value();
        Percentile percentile;
        switch (xrt_metric.unit_of_measure()) {
          case xrt::MetricValues::NUMBER:
            percentile.unit_of_measure = Percentile::UnitOfMeaure::kNumber;
            break;
          case xrt::MetricValues::TIME:
            percentile.unit_of_measure = Percentile::UnitOfMeaure::kTime;
            break;
          case xrt::MetricValues::BYTES:
            percentile.unit_of_measure = Percentile::UnitOfMeaure::kBytes;
            break;
          default:
            TF_LOG(FATAL) << "Invalid unit of measure";
            break;
        }
        percentile.start_nstime = xrt_percentile.start_nstime();
        percentile.end_nstime = xrt_percentile.end_nstime();
        percentile.min_value = xrt_percentile.min_value();
        percentile.max_value = xrt_percentile.max_value();
        percentile.mean = xrt_percentile.mean();
        percentile.stddev = xrt_percentile.stddev();
        percentile.num_samples = xrt_percentile.num_samples();
        percentile.total_samples = xrt_percentile.total_samples();
        percentile.accumulator = xrt_percentile.accumulator();
        for (auto& xrt_point : xrt_percentile.points()) {
          percentile.points.push_back(
              Percentile::Point{xrt_point.percentile(), xrt_point.value()});
        }
        metric.percentile = std::move(percentile);
      } else if (xrt_metric.values_oneof_case() ==
                 xrt::MetricValues::kInt64Value) {
        metric.int64_value = xrt_metric.int64_value();
      } else {
        continue;
      }

      std::string metric_name;
      auto it = metric_remap->find(xrt_metric.name());
      if (it != metric_remap->end()) {
        metric_name = it->second;
      } else {
        metric_name = xrt_metric.name();
      }
      if (options_.workers_map.size() > 1) {
        absl::StrAppend(&metric_name, ".", worker_target.first.name, ".",
                        worker_target.first.task_no);
      }
      metrics_data.emplace(std::move(metric_name), std::move(metric));
    }
  }
  return metrics_data;
}

void XrtComputationClient::InitSession(XrtSession* session) const {
  struct InitNode {
    int count;
    const XrtSession::CachedNode& (XrtComputationClient::*node_ctor)(
        XrtSession*, const tensorflow::Scope&, const std::string&) const;
  } const init_nodes[] = {
      {16, &XrtComputationClient::GetCompileNode},
      {16, &XrtComputationClient::GetExecuteNode},
      {16, &XrtComputationClient::GetExecuteChainedNode},
      {16, &XrtComputationClient::GetReadNode},
      {16, &XrtComputationClient::GetReleaseAllocationHandleNode},
      {16, &XrtComputationClient::GetReleaseCompileHandleNode},
      {16, &XrtComputationClient::GetSubTupleNode},
  };
  auto devices = GetLocalDevices();
  for (auto& device : devices) {
    // HACK: The XRT ops on the remote GRPC service has only recently been
    // enabled, so until TF 1.14 is out, we cannot add XRT ops on CPU.
    // If there is only one device, even if CPU, this is the local session,
    // which carries the XRT op (as we include them in the BUILD).
    if (device.compare(0, 4, "CPU:") == 0 && devices.size() > 1) {
      continue;
    }
    const std::string& xrt_device = SwiftDeviceToXrtDevice(device);
    tensorflow::Scope device_scope = session->root()->WithDevice(xrt_device);
    for (auto& init : init_nodes) {
      for (int i = 0; i < init.count; ++i) {
        (this->*init.node_ctor)(session, device_scope, device);
      }
    }
  }
  session->Reset();
}

const XrtSession::CachedNode& XrtComputationClient::GetCompileNode(
    XrtSession* session, const tensorflow::Scope& scope,
    const std::string& device) const {
  static const std::string op_name("XrtCompile");  // NOLINT
  XrtSession::NodeCache* cache =
      session->GetNodeCache(XrtSession::GetCacheKey(op_name, device));
  if (cache->Empty()) {
    XLA_COUNTER("XrtCompile_Empty", 1);
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(scope, tensorflow::DT_STRING)});
    cache->Add(std::make_shared<XrtSession::CachedNode>(
        tensorflow::ops::XRTCompile(scope, holders[0]).handle, holders));
  }
  return cache->Get();
}

const XrtSession::CachedNode& XrtComputationClient::GetExecuteNode(
    XrtSession* session, const tensorflow::Scope& scope,
    const std::string& device) const {
  static const std::string op_name("XrtExecute");  // NOLINT
  XrtSession::NodeCache* cache =
      session->GetNodeCache(XrtSession::GetCacheKey(op_name, device));
  if (cache->Empty()) {
    XLA_COUNTER("XrtExecute_Empty", 1);
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(scope, tensorflow::DT_INT64),
         tensorflow::ops::Placeholder(scope, tensorflow::DT_STRING),
         tensorflow::ops::Placeholder(
             scope, tensorflow::DT_INT64,
             tensorflow::ops::Placeholder::Shape({-1}))});
    cache->Add(std::make_shared<XrtSession::CachedNode>(
        tensorflow::ops::XRTExecute(scope, holders[0], holders[1],
                                    {tensorflow::Output(holders[2])}),
        holders));
  }
  return cache->Get();
}

const XrtSession::CachedNode& XrtComputationClient::GetExecuteChainedNode(
    XrtSession* session, const tensorflow::Scope& scope,
    const std::string& device) const {
  static const std::string op_name("XrtExecuteChained");  // NOLINT
  XrtSession::NodeCache* cache =
      session->GetNodeCache(XrtSession::GetCacheKey(op_name, device));
  if (cache->Empty()) {
    XLA_COUNTER("XrtExecuteChained_Empty", 1);
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(scope, tensorflow::DT_STRING),
         tensorflow::ops::Placeholder(scope, tensorflow::DT_STRING)});
    cache->Add(std::make_shared<XrtSession::CachedNode>(
        tensorflow::ops::XRTExecuteChained(scope, holders[0], holders[1]),
        holders));
  }
  return cache->Get();
}

const XrtSession::CachedNode& XrtComputationClient::GetReadNode(
    XrtSession* session, const tensorflow::Scope& scope,
    const std::string& device) const {
  static const std::string op_name("XrtRead");  // NOLINT
  XrtSession::NodeCache* cache =
      session->GetNodeCache(XrtSession::GetCacheKey(op_name, device));
  if (cache->Empty()) {
    XLA_COUNTER("XrtRead_Empty", 1);
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(scope, tensorflow::DT_INT64)});
    cache->Add(std::make_shared<XrtSession::CachedNode>(
        tensorflow::ops::XRTReadLiteral(scope, holders[0]), holders));
  }
  return cache->Get();
}

const XrtSession::CachedNode& XrtComputationClient::GetAllocateNode(
    XrtSession* session, const tensorflow::Scope& scope,
    const std::string& device, const Shape& shape) const {
  // Create the proper key for the allocation node. Since the node has shape and
  // layouts attributes, these need to be included within the key.
  std::stringstream ss;
  ss << "XRTAllocateFromTensor(" << shape << ")";
  XrtSession::NodeCache* cache =
      session->GetNodeCache(XrtSession::GetCacheKey(ss.str(), device));
  if (cache->Empty()) {
    XLA_COUNTER("XRTAllocateFromTensor_Empty", 1);
    tensorflow::TensorShape tensor_shape(shape.dimensions());
    tensorflow::TensorShape equiv_tensor_shape =
        MakeEquivalentTensorShape(shape);
    std::vector<int> layout(shape.layout().minor_to_major().begin(),
                            shape.layout().minor_to_major().end());
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(
            scope, XlaTypeToDataType(shape.element_type()),
            tensorflow::ops::Placeholder::Shape(equiv_tensor_shape))});
    tensorflow::ops::XRTAllocateFromTensor::Attrs alloc_attrs =
        tensorflow::ops::XRTAllocateFromTensor::Layouts(layout);
    cache->Add(std::make_shared<XrtSession::CachedNode>(
        tensorflow::ops::XRTAllocateFromTensor(scope, {holders[0].output},
                                               {tensor_shape}, alloc_attrs),
        holders));
  }
  return cache->Get();
}

const XrtSession::CachedNode&
XrtComputationClient::GetReleaseAllocationHandleNode(
    XrtSession* session, const tensorflow::Scope& scope,
    const std::string& device) const {
  static const std::string op_name("XrtReleaseAllocationHandle");  // NOLINT
  XrtSession::NodeCache* cache =
      session->GetNodeCache(XrtSession::GetCacheKey(op_name, device));
  if (cache->Empty()) {
    XLA_COUNTER("XrtReleaseAllocationHandle_Empty", 1);
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(scope, tensorflow::DT_INT64)});
    cache->Add(std::make_shared<XrtSession::CachedNode>(
        tensorflow::ops::XRTReleaseAllocationHandle(scope, holders[0]),
        holders));
  }
  return cache->Get();
}

const XrtSession::CachedNode& XrtComputationClient::GetReleaseCompileHandleNode(
    XrtSession* session, const tensorflow::Scope& scope,
    const std::string& device) const {
  static const std::string op_name("XrtReleaseCompileHandle");  // NOLINT
  XrtSession::NodeCache* cache =
      session->GetNodeCache(XrtSession::GetCacheKey(op_name, device));
  if (cache->Empty()) {
    XLA_COUNTER("XrtReleaseCompileHandle_Empty", 1);
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(scope, tensorflow::DT_INT64)});
    cache->Add(std::make_shared<XrtSession::CachedNode>(
        tensorflow::ops::XRTReleaseCompilationHandle(scope, holders[0]),
        holders));
  }
  return cache->Get();
}

const XrtSession::CachedNode& XrtComputationClient::GetSubTupleNode(
    XrtSession* session, const tensorflow::Scope& scope,
    const std::string& device) const {
  static const std::string op_name("XrtSubTuple");  // NOLINT
  XrtSession::NodeCache* cache =
      session->GetNodeCache(XrtSession::GetCacheKey(op_name, device));
  if (cache->Empty()) {
    XLA_COUNTER("XrtSubTuple_Empty", 1);
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(scope, tensorflow::DT_INT64),
         tensorflow::ops::Placeholder(
             scope, tensorflow::DT_INT32,
             tensorflow::ops::Placeholder::Shape({1}))});
    cache->Add(std::make_shared<XrtSession::CachedNode>(
        tensorflow::ops::XRTSubTuple(scope, holders[0], holders[1]), holders));
  }
  return cache->Get();
}

tensorflow::DataType XrtComputationClient::XlaTypeToDataType(
    PrimitiveType dtype) {
  switch (dtype) {
    case PrimitiveType::PRED:
      return tensorflow::DT_BOOL;
    case PrimitiveType::S8:
      return tensorflow::DT_INT8;
    case PrimitiveType::U8:
      return tensorflow::DT_UINT8;
    case PrimitiveType::S16:
      return tensorflow::DT_INT16;
    case PrimitiveType::U16:
      return tensorflow::DT_UINT16;
    case PrimitiveType::S32:
      return tensorflow::DT_INT32;
    case PrimitiveType::U32:
      return tensorflow::DT_UINT32;
    case PrimitiveType::S64:
      return tensorflow::DT_INT64;
    case PrimitiveType::U64:
      return tensorflow::DT_UINT64;
    case PrimitiveType::F32:
      return tensorflow::DT_FLOAT;
    case PrimitiveType::F64:
      return tensorflow::DT_DOUBLE;
    case PrimitiveType::BF16:
      return tensorflow::DT_BFLOAT16;
    case PrimitiveType::F16:
      return tensorflow::DT_HALF;
    case PrimitiveType::C64:
      return tensorflow::DT_COMPLEX64;
    case PrimitiveType::C128:
      return tensorflow::DT_COMPLEX128;
    default:
      break;
  }
  XLA_ERROR() << "Unable to convert XLA type " << dtype
              << " to tensorflow DataType";
}

tensorflow::TensorShape XrtComputationClient::MakeEquivalentTensorShape(
    const Shape& shape) {
  Shape eqiv_shape =
      ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(shape);
  return tensorflow::TensorShape(eqiv_shape.dimensions());
}

std::vector<std::vector<ComputationClient::DataPtr>>
XrtComputationClient::BuildParallelArguments(
    absl::Span<const DataPtr> arguments) {
  std::vector<std::vector<DataPtr>> para_arguments(1);
  para_arguments[0].insert(para_arguments[0].end(), arguments.begin(),
                           arguments.end());
  return para_arguments;
}

tensorflow::ConfigProto XrtComputationClient::CreateConfigProto(
    const Options& options) {
  static const std::string* const grpc_proto = new std::string("grpc://");
  tensorflow::ConfigProto config;
  if (options.workers_map.size() > 1) {
    tensorflow::ClusterDef* cluster_def = config.mutable_cluster_def();
    std::map<std::string, tensorflow::JobDef*> jobs;
    for (auto& worker_target : options.workers_map) {
      auto it = jobs.find(worker_target.first.name);
      if (it == jobs.end()) {
        tensorflow::JobDef* job = cluster_def->add_job();
        job->set_name(worker_target.first.name);
        it = jobs.emplace(worker_target.first.name, job).first;
      }
      tensorflow::JobDef* job = it->second;
      (*job->mutable_tasks())[worker_target.first.task_no] =
          StripPrefix(worker_target.second, *grpc_proto);
    }
  }
  return config;
}

XrtComputationClient::Worker XrtComputationClient::ParseWorker(
    const std::string& worker) {
  std::vector<std::string> parts = absl::StrSplit(worker, ':');
  XLA_CHECK(parts.size() == 1 || parts.size() == 2) << worker;
  return parts.size() == 1 ? Worker(parts[0], 0)
                           : Worker(parts[0], std::stoi(parts[1]));
}

std::string XrtComputationClient::GetLocalTarget(const Options& options) {
  std::string local_worker = sys_util::GetEnvString(env::kEnvLocalWorker, "");
  std::string local_target;
  if (!local_worker.empty()) {
    XrtComputationClient::Worker worker = ParseWorker(local_worker);
    if (worker.name == kLocalService) {
      auto it = options.workers_map.find(worker);
      if (it != options.workers_map.end()) {
        local_target = it->second;
      }
    }
  }
  return local_target;
}

void XrtComputationClient::MaybeCreateLocalService(const Options& options) {
  std::string grpc_root("grpc://");
  std::string local_worker = sys_util::GetEnvString(env::kEnvLocalWorker, "");
  XrtComputationClient::Worker worker("", -1);
  if (!local_worker.empty()) {
    worker = ParseWorker(local_worker);
  }
  int task_index = -1;
  std::string job_name;
  std::vector<std::string> hosts;
  for (auto& worker_target : options.workers_map) {
    if (worker_target.first.name == kLocalService &&
        worker_target.second.compare(0, grpc_root.size(), grpc_root) == 0) {
      hosts.push_back(worker_target.second.substr(grpc_root.size()));
      if (worker.task_no < 0 || worker_target.first == worker) {
        XLA_CHECK_EQ(task_index, -1)
            << "Multiple workers matching the local one: '" << local_worker
            << "'";
        job_name = worker_target.first.name;
        task_index = worker_target.first.task_no;
      }
    }
  }
  if (task_index >= 0 && !job_name.empty()) {
    std::string cluster_spec =
        absl::StrCat(job_name, "|", absl::StrJoin(hosts, ";"));
    TF_VLOG(2) << "Local Service Cluster Spec: " << cluster_spec;
    XrtLocalService* service =
        new XrtLocalService(cluster_spec, job_name, task_index);
    service->Start();
  }
}

std::string XrtComputationClient::GetMultiProcessingDevice() {
  return sys_util::GetEnvString(env::kEnvMpDevice, "");
}

swift_xla::Device XrtComputationClient::GetDefaultDeviceStruct() const {
  return *swift_xla::GetDefaultDevice();
}

}  // namespace xla
