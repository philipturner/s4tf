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

#include "tensorflow/compiler/tf2xla/xla_tensor/layout_manager.h"

#include <algorithm>
#include <exception>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>

#include "absl/container/node_hash_map.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/tf_logging.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace swift_xla {
namespace {

class LayoutManager {
 public:
  static LayoutManager* Get() {
    static LayoutManager* mgr = new LayoutManager();
    return mgr;
  }

  const std::vector<int64_t>* GetLayout(
      absl::Span<const int64_t> dimensions) const {
    auto it = layouts_.find(dimensions);
    return it != layouts_.end() ? &it->second->layout : nullptr;
  }

 private:
  struct LayoutEntry {
    std::vector<int64_t> dimensions;
    std::vector<int64_t> layout;
  };

  struct DimensionsHasher {
    size_t operator()(const absl::Span<const int64_t>& dimensions) const {
      return xla::util::HashReduce(xla::util::MHash(dimensions));
    }
  };

  using LayoutMap =
      absl::node_hash_map<absl::Span<const int64_t>,
                          std::shared_ptr<LayoutEntry>, DimensionsHasher>;

  LayoutManager() { PopulateLayouts(); }

  // TODO(asuhan): Return status.
  void PopulateLayouts() {
    // Layouts: SHAPE=LAYOUT;...
    // SHAPE: INT,...
    // LAYOUT: INT,...
    std::string layouts_env = xla::sys_util::GetEnvString("XLA_LAYOUTS", "");
    if (!layouts_env.empty()) {
      std::vector<std::string> layouts = absl::StrSplit(layouts_env, ';');
      for (const auto& layout_str : layouts) {
        std::vector<std::string> parts = absl::StrSplit(layout_str, '=');
        XLA_CHECK_EQ(parts.size(), 2) << layout_str;

        auto entry = std::make_shared<LayoutEntry>();
        entry->dimensions = ParseIntList(parts[0]);
        entry->layout = ParseLayout(parts[1], entry->dimensions.size());
        layouts_.emplace(entry->dimensions, entry);

        TF_VLOG(2) << "Registering layout " << parts[1] << " for shape "
                   << parts[0];
      }
    }
  }

  static std::vector<int64_t> ParseIntList(const std::string& list_str) {
    std::vector<std::string> parts = absl::StrSplit(list_str, ',');
    std::vector<int64_t> ints;
    for (const auto& int_str : parts) {
      ints.push_back(std::stol(int_str));
    }
    return ints;
  }

  static std::vector<int64_t> ParseLayout(const std::string& list_str,
                                             int64_t rank) {
    std::vector<int64_t> ints = ParseIntList(list_str);
    XLA_CHECK_EQ(ints.size(), rank) << list_str;
    std::set<int64_t> unique_ints;
    for (auto dim : ints) {
      XLA_CHECK_GE(dim, 0) << list_str;
      XLA_CHECK_LT(dim, rank) << list_str;
      unique_ints.insert(dim);
    }
    XLA_CHECK_EQ(unique_ints.size(), rank) << list_str;
    return ints;
  }

  LayoutMap layouts_;
};

double PaddingFactor(int64_t size, int padding) {
  int rem = static_cast<int>(size % padding);
  return 1.0 + (rem > 0 ? static_cast<double>(padding - rem) /
                              static_cast<double>(size)
                        : 0.0);
}

xla::Shape MakeShapeWithSortedLayout(absl::Span<const int64_t> dimensions,
                                     xla::PrimitiveType type) {
  // Place bigger dimensions on most minor layout locations.
  std::vector<int64_t> layout =
      xla::util::Iota<int64_t>(dimensions.size(), dimensions.size() - 1, -1);
  std::sort(layout.begin(), layout.end(), [&](int64_t a, int64_t b) {
    return dimensions[a] > dimensions[b];
  });
  return xla::ShapeUtil::MakeShapeWithLayout(type, dimensions, layout);
}

xla::Shape* SetDynamicDimensions(xla::Shape* shape,
                                 absl::Span<const bool> dynamic_dimensions) {
  if (!dynamic_dimensions.empty()) {
    XLA_CHECK_EQ(dynamic_dimensions.size(), shape->rank());
    for (size_t i = 0; i < dynamic_dimensions.size(); ++i) {
      shape->set_dynamic_dimension(i, dynamic_dimensions[i]);
    }
  }
  return shape;
}

xla::Shape MakeTpuShape(absl::Span<const int64_t> dimensions,
                        absl::Span<const bool> dynamic_dimensions,
                        xla::PrimitiveType type) {
  static double max_padding_factor =
      xla::sys_util::GetEnvDouble("XLA_MAX_PADDING_FACTOR", 1.25);
  xla::Shape shape;
  if (PaddingFactor(dimensions[dimensions.size() - 1], 128) *
          PaddingFactor(dimensions[dimensions.size() - 2], 8) <
      max_padding_factor) {
    shape = xla::ShapeUtil::MakeShapeWithDescendingLayout(type, dimensions);
  } else {
    shape = MakeShapeWithSortedLayout(dimensions, type);
  }
  SetDynamicDimensions(&shape, dynamic_dimensions);
  return shape;
}

xla::Shape MakeShapeWithLayout(xla::PrimitiveType type,
                               absl::Span<const int64_t> dimensions,
                               absl::Span<const bool> dynamic_dimensions,
                               absl::Span<const int64_t> layout) {
  xla::Shape shape =
      xla::ShapeUtil::MakeShapeWithLayout(type, dimensions, layout);
  SetDynamicDimensions(&shape, dynamic_dimensions);
  return shape;
}

}  // namespace

xla::Shape MakeSwiftTensorLayout(absl::Span<const int64_t> dimensions,
                                 absl::Span<const bool> dynamic_dimensions,
                                 xla::PrimitiveType type) {
  xla::Shape shape =
      xla::ShapeUtil::MakeShapeWithDescendingLayout(type, dimensions);
  SetDynamicDimensions(&shape, dynamic_dimensions);
  return shape;
}

xla::Shape MakeArrayShapeFromDimensions(
    absl::Span<const int64_t> dimensions,
    absl::Span<const bool> dynamic_dimensions, xla::PrimitiveType type,
    DeviceType device_type) {
  auto layout_ptr = LayoutManager::Get()->GetLayout(dimensions);
  if (layout_ptr != nullptr) {
    return MakeShapeWithLayout(type, dimensions, dynamic_dimensions,
                               *layout_ptr);
  }
  if (dimensions.size() > 1 && device_type == DeviceType::TPU) {
    return MakeTpuShape(dimensions, dynamic_dimensions, type);
  }
  return MakeSwiftTensorLayout(dimensions, dynamic_dimensions, type);
}

}  // namespace swift_xla
