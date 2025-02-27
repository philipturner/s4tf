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

#include "tensorflow/compiler/tf2xla/xla_tensor/convert_ops.h"

#include <climits>

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/tensor_util.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace swift_xla {
namespace {

xla::XlaOp ExplicitBooleanConvert(xla::XlaOp op, xla::PrimitiveType from) {
  xla::XlaOp zero = xla::Zero(op.builder(), from);
  return xla::Ne(op, zero);
}

xla::XlaOp CreateRawMask(xla::XlaOp op, xla::PrimitiveType type,
                         int64_t to_size, int64_t raw_to_size) {
  xla::uint64 mask_value =
      (static_cast<xla::uint64>(1) << raw_to_size * CHAR_BIT) - 1;
  xla::XlaOp mask = XlaHelpers::ScalarValue(mask_value, type, op.builder());
  if (xla::primitive_util::IsSignedIntegralType(type)) {
    // Sign extend the truncation.
    xla::XlaOp shift = XlaHelpers::ScalarValue<xla::int32>(
        (to_size - raw_to_size) * CHAR_BIT, op.builder());
    mask = (mask << shift) >> shift;
  }
  const xla::Shape& op_shape = XlaHelpers::ShapeOfXlaOp(op);
  return op_shape.rank() > 0 ? xla::Broadcast(mask, op_shape.dimensions())
                             : mask;
}

xla::XlaOp ConvertData(xla::XlaOp op, xla::PrimitiveType to,
                       xla::PrimitiveType raw_to) {
  if (!xla::primitive_util::IsIntegralType(to) ||
      !xla::primitive_util::IsIntegralType(raw_to)) {
    return op;
  }
  int64_t to_size = xla::ShapeUtil::ByteSizeOfPrimitiveType(to);
  int64_t raw_to_size = xla::ShapeUtil::ByteSizeOfPrimitiveType(raw_to);
  XLA_CHECK_GE(to_size, raw_to_size);
  if (to_size == raw_to_size) {
    return op;
  }
  xla::XlaOp mask = CreateRawMask(op, to, to_size, raw_to_size);
  return op & mask;
}

}  // namespace

xla::XlaOp ConvertTo(xla::XlaOp op, xla::PrimitiveType from,
                     xla::PrimitiveType to, const Device* device) {
  if (from == to) {
    return op;
  }
  if (GetDeviceOrCurrent(device).hw_type != DeviceType::TPU) {
    return xla::ConvertElementType(op, to);
  }
  switch (from) {
    case xla::PrimitiveType::PRED:
    case xla::PrimitiveType::S8:
    case xla::PrimitiveType::U8:
    case xla::PrimitiveType::S16:
    case xla::PrimitiveType::U16:
    case xla::PrimitiveType::S32:
    case xla::PrimitiveType::U32:
    case xla::PrimitiveType::BF16:
    case xla::PrimitiveType::F32:
      return xla::ConvertElementType(op, to);
    case xla::PrimitiveType::S64:
    case xla::PrimitiveType::U64: {
      switch (to) {
        case xla::PrimitiveType::PRED:
          return ExplicitBooleanConvert(op, from);
        default:
          return xla::ConvertElementType(op, to);
      }
      break;
    }
    default:
      XLA_ERROR() << "Unsupported XLA type " << from;
  }
}

xla::XlaOp ConvertToRaw(xla::XlaOp op, xla::PrimitiveType from,
                        xla::PrimitiveType to, xla::PrimitiveType raw_to,
                        const Device* device) {
  xla::XlaOp result = ConvertTo(op, from, to, device);
  return to == raw_to ? result : ConvertData(result, to, raw_to);
}

xla::XlaOp ConvertToNumeric(xla::XlaOp op, xla::PrimitiveType from) {
  if (from == xla::PrimitiveType::PRED) {
    Device xla_device = GetCurrentDevice();
    op = ConvertTo(op, from,
                   GetDevicePrimitiveType(xla::PrimitiveType::U8, &xla_device),
                   &xla_device);
  }
  return op;
}

xla::XlaOp ConvertToNumeric(xla::XlaOp op) {
  return ConvertToNumeric(op, XlaHelpers::TypeOfXlaOp(op));
}

xla::XlaOp CastToScalarType(xla::XlaOp input,
                            c10::optional<at::ScalarType> dtype) {
  if (dtype) {
    return ConvertTo(input, XlaHelpers::TypeOfXlaOp(input),
                     MakeXlaPrimitiveType(*dtype, /*device=*/nullptr),
                     /*device=*/nullptr);
  } else {
    return ConvertToNumeric(input, XlaHelpers::TypeOfXlaOp(input));
  }
}

xla::XlaOp MaybeConvertTo(xla::XlaOp input, xla::PrimitiveType type) {
  return XlaHelpers::TypeOfXlaOp(input) != type
             ? xla::ConvertElementType(input, type)
             : input;
}

}  // namespace swift_xla
