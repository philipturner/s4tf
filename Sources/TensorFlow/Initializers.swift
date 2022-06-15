//// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
////
//// Licensed under the Apache License, Version 2.0 (the "License");
//// you may not use this file except in compliance with the License.
//// You may obtain a copy of the License at
////
////     http://www.apache.org/licenses/LICENSE-2.0
////
//// Unless required by applicable law or agreed to in writing, software
//// distributed under the License is distributed on an "AS IS" BASIS,
//// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//// See the License for the specific language governing permissions and
//// limitations under the License.

import _Differentiation

//#if !COMPILING_TENSORFLOW_STDLIB_MODULE
//  import Tensor
//#endif

//extension Tensor {
//  /// Creates a tensor with the specified shape and a single, repeated scalar value.
//  ///
//  /// - Parameters:
//  ///   - shape: The dimensions of the tensor.
//  ///   - repeatedValue: The scalar value to repeat.
////  @inlinable
////  @available(*, deprecated, renamed: "init(repeating:shape:)")
////  public init(shape: TensorShape, repeating repeatedValue: Scalar) {
////    self.init(repeating: repeatedValue, shape: shape)
////  }
//
//  /// Creates a tensor with the specified shape and a single, repeated scalar value.
//  ///
//  /// - Parameters:
//  ///   - repeatedValue: The scalar value to repeat.
//  ///   - shape: The dimensions of the tensor.
////  @inlinable
////  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
////  public init(
////    repeating repeatedValue: Scalar, shape: TensorShape,
////    on device: Device = .default
////  ) {
////    self = _Raw.fill(
////      dims: Tensor<Int32>(shape.dimensions.map(Int32.init), on: device),
////      value: Tensor(repeatedValue, on: device))
////  }
//
//  /// Creates a tensor by broadcasting the given scalar to a given rank with
////  /// all dimensions being 1.
////  @inlinable
////  // @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
////  public init(broadcasting scalar: Scalar, rank: Int, on device: Device = .default) {
////    self = Tensor(scalar, on: device).reshaped(to: TensorShape(repeating: 1, count: rank))
////  }
//
//  /// Creates a tensor of shape `[4]` from a 4-tuple.
//  /// - Note: This is intended for internal use, for example, to initialize a
////  ///   tensor attribute from `convolved2D`'s `strides` argument.
////  @inlinable
////  internal init(_ scalars: (Scalar, Scalar, Scalar, Scalar), on device: Device = .default) {
////    fatalError()
//////    self.init([scalars.0, scalars.1, scalars.2, scalars.3], on: device)
////  }
//}

//extension Tensor where Scalar: TensorFlowFloatingPoint {
//  @inlinable
////  @derivative(of: init(repeating:shape:on:))
//  static func _vjpInit(
//    repeating repeatedValue: __owned Scalar,
//    shape: __owned TensorShape,
//    on device: Device
//  ) -> (value: Tensor, pullback: (Tensor) -> Scalar) {
//    fatalError()
////    return (
////      Tensor(repeating: repeatedValue, shape: shape, on: device),
////      {
////        $0.sum().scalarized()
////      }
////    )
//  }
//}

//===------------------------------------------------------------------------------------------===//
// Casting
//===------------------------------------------------------------------------------------------===//

//extension Tensor where Scalar: Numeric {
//  /// Perform an element-wise type conversion from a `Bool` tensor.
////  @inlinable
////  public init(_ other: Tensor<Bool>) {
////    self = _Raw.cast(other)
////  }
//
//  /// Perform an element-wise conversion from another `Tensor`.
////  @inlinable
////  @differentiable(reverse where Scalar: TensorFlowFloatingPoint, OtherScalar: TensorFlowFloatingPoint)
////  public init<OtherScalar: Numeric>(_ other: Tensor<OtherScalar>) {
////    fatalError()
//////    self = _Raw.cast(other)
////  }
//}

//extension Tensor where Scalar: TensorFlowFloatingPoint {
//  @inlinable
//  @derivative(of: init(_:))
//  static func _vjpCast<OtherScalar: TensorFlowFloatingPoint>(
//    _ other: __owned Tensor<OtherScalar>
//  ) -> (value: Tensor, pullback: (Tensor) -> Tensor<OtherScalar>) {
//    (Tensor(other), { v in Tensor<OtherScalar>(v) })
//  }
//}

//===------------------------------------------------------------------------------------------===//
// Numeric
//===------------------------------------------------------------------------------------------===//

extension Tensor where Scalar: Numeric {
  /// Creates a tensor with all scalars set to zero.
  ///
  /// - Parameter shape: Shape of the tensor.
//  @inlinable
  public init(zeros shape: TensorShape, on device: Device = .default) {
    fatalError()
//    self.init(repeating: 0, shape: shape, on: device)
  }

  /// Creates a tensor with all scalars set to one.
  ///
  /// - Parameter shape: Shape of the tensor.
//  @inlinable
  public init(ones shape: TensorShape, on device: Device = .default) {
    fatalError()
//    self.init(repeating: 1, shape: shape, on: device)
  }

  /// Creates a tensor with all scalars set to zero that has the same shape and type as the provided
  /// tensor.
//  ///
//  /// - Parameter other: Tensor whose shape and data type to use.
//  @inlinable
//  public init(zerosLike other: Tensor) {
//    self = _Raw.zerosLike(other)
//  }

  /// Creates a tensor with all scalars set to one that has the same shape and type as the provided
  /// tensor.
//  ///
//  /// - Parameter other: Tensor whose shape and data type to use.
//  @inlinable
//  public init(onesLike other: Tensor) {
//    self = _Raw.onesLike(other)
//  }

  /// Creates a 1-D tensor representing a sequence from a starting value to, but not including,
  /// an end value, stepping by the specified amount.
  ///
  /// - Parameters:
  ///   - start: The starting value to use for the sequence. If the sequence
  ///     contains any values, the first one is `start`.
  ///   - end: An end value to limit the sequence. `end` is never an element of
  ///     the resulting sequence.
  ///   - stride: The amount to step by with each iteration. `stride` must be
//  ///     positive.
//  @inlinable
//  public init(
//    rangeFrom start: Scalar, to end: Scalar, stride: Scalar,
//    on device: Device = .default
//  ) {
//    self = _Raw.range(
//      start: Tensor(start, on: device), limit: Tensor(end, on: device),
//      delta: Tensor(stride, on: device))
//  }

  /// Creates a 1-D tensor representing a sequence from a starting value to, but not including, an
  /// end value, stepping by the specified amount.
  ///
  /// - Parameters:
  ///   - start: The starting value to use for the sequence. If the sequence contains any values,
  ///     the first one is `start`.
  ///   - end: An end value to limit the sequence. `end` is never an element of the resulting
  ///     sequence.
//  ///   - stride: The amount to step by with each iteration. `stride` must be positive.
//  @inlinable
//  public init(rangeFrom start: Tensor<Scalar>, to end: Tensor<Scalar>, stride: Tensor<Scalar>) {
//    self = _Raw.range(start: start, limit: end, delta: stride)
//  }

  /// Creates a one-hot tensor at given indices. The locations represented by
  /// `indices` take value `onValue` (`1` by default), while all other locations
  /// take value `offValue` (`0` by default). If the input `indices` is rank
  /// `n`, the new tensor will have rank `n+1`. The new axis is created at
  /// dimension `axis` (by default, the new axis is appended at the end).
  ///
  /// If `indices` is a scalar, the new tensor's shape will be a vector of
  /// length `depth`.
  ///
  /// If `indices` is a vector of length `features`, the output shape will be:
  ///     features x depth, if axis == -1
  ///     depth x features, if axis == 0
  ///
  /// If `indices` is a matrix (batch) with shape `[batch, features]`, the
  /// output shape will be:
  ///     batch x features x depth, if axis == -1
  ///     batch x depth x features, if axis == 1
  ///     depth x batch x features, if axis == 0
  ///
  /// - Parameters:
  ///   - indices: A `Tensor` of indices.
  ///   - depth: A scalar defining the depth of the one hot dimension.
  ///   - onValue: A scalar defining the value at the location referred to by
  ///     some index in `indices`.
  ///   - offValue: A scalar defining the value at a location that is not
  ///     referred to by any index in `indices`.
  ///   - axis: The axis to fill. The default is `-1`, a new inner-most axis.
//  @inlinable
//  public init(
//    oneHotAtIndices indices: Tensor<Int32>,
//    depth: Int,
//    onValue: Scalar = 1,
//    offValue: Scalar = 0,
//    axis: Int = -1
//  ) {
//    let device = indices.device
//    self = _Raw.oneHot(
//      indices: indices,
//      depth: Tensor<Int32>(Int32(depth), on: device),
//      onValue: Tensor(onValue, on: device),
//      offValue: Tensor(offValue, on: device),
//      axis: Int64(axis))
//  }
}
