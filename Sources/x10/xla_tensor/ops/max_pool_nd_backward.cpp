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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/max_pool_nd_backward.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/pooling.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& grad_output, const Value& input,
                           int64_t spatial_dim_count,
                           absl::Span<const int64_t> kernel_size,
                           absl::Span<const int64_t> stride,
                           absl::Span<const int64_t> padding,
                           bool ceil_mode) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 2);
    return BuildMaxPoolNdBackward(/*out_backprop=*/operands[0],
                                  /*input=*/operands[1], spatial_dim_count,
                                  kernel_size, stride, padding, ceil_mode);
  };
  return InferOutputShape({grad_output.shape(), input.shape()},
                          lower_for_shape_fn);
}

c10::Symbol MaxPoolNdBackwardSymbol(int64_t spatial_dim_count) {
  switch (spatial_dim_count) {
    case 2:
      return at::aten::max_pool2d_with_indices_backward;
    case 3:
      return at::aten::max_pool3d_with_indices_backward;
    default:
      XLA_ERROR() << "Invalid number of spatial dimensions: "
                  << spatial_dim_count;
  }
}

}  // namespace

MaxPoolNdBackward::MaxPoolNdBackward(
    const Value& grad_output, const Value& input, int64_t spatial_dim_count,
    std::vector<int64_t> kernel_size, std::vector<int64_t> stride,
    std::vector<int64_t> padding, bool ceil_mode)
    : Node(ir::OpKind(MaxPoolNdBackwardSymbol(spatial_dim_count)),
           {grad_output, input},
           [&]() {
             return NodeOutputShape(grad_output, input, spatial_dim_count,
                                    kernel_size, stride, padding, ceil_mode);
           },
           /*num_outputs=*/1,
           xla::util::MHash(spatial_dim_count, kernel_size, stride, padding,
                            ceil_mode)),
      spatial_dim_count_(spatial_dim_count),
      kernel_size_(std::move(kernel_size)),
      stride_(std::move(stride)),
      padding_(std::move(padding)),
      ceil_mode_(ceil_mode) {}

NodePtr MaxPoolNdBackward::Clone(OpList operands) const {
  return MakeNode<MaxPoolNdBackward>(operands.at(0), operands.at(1),
                                     spatial_dim_count_, kernel_size_, stride_,
                                     padding_, ceil_mode_);
}

XlaOpVector MaxPoolNdBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp output = BuildMaxPoolNdBackward(
      /*out_backprop=*/grad_output, /*input=*/input, spatial_dim_count_,
      kernel_size_, stride_, padding_, ceil_mode_);
  return ReturnOp(output, loctx);
}

std::string MaxPoolNdBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", spatial_dim_count=" << spatial_dim_count_
     << ", kernel_size=(" << absl::StrJoin(kernel_size_, ", ") << "), stride=("
     << absl::StrJoin(stride_, ", ") << "), padding=("
     << absl::StrJoin(padding_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
