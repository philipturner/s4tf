// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import _Differentiation

#if !COMPILING_TENSORFLOW_STDLIB_MODULE
  import Tensor
#endif

extension Tensor {
  /// Creates a tensor with the specified shape and a single, repeated scalar value.
  ///
  /// - Parameters:
  ///   - shape: The dimensions of the tensor.
  ///   - repeatedValue: The scalar value to repeat.
  @inlinable
  @available(*, deprecated, renamed: "init(repeating:shape:)")
  public init(shape: TensorShape, repeating repeatedValue: Scalar) {
    self.init(repeating: repeatedValue, shape: shape)
  }

  /// Creates a tensor with the specified shape and a single, repeated scalar value.
  ///
  /// - Parameters:
  ///   - repeatedValue: The scalar value to repeat.
  ///   - shape: The dimensions of the tensor.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public init(
    repeating repeatedValue: Scalar, shape: TensorShape,
    on device: Device = .default
  ) {
    self = _Raw.fill(
      dims: Tensor<Int32>(shape.dimensions.map(Int32.init), on: device),
      value: Tensor(repeatedValue, on: device))
  }

  /// Creates a tensor by broadcasting the given scalar to a given rank with
  /// all dimensions being 1.
  @inlinable
  // @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public init(broadcasting scalar: Scalar, rank: Int, on device: Device = .default) {
    self = Tensor(scalar, on: device).reshaped(to: TensorShape(repeating: 1, count: rank))
  }

  /// Creates a tensor of shape `[4]` from a 4-tuple.
  /// - Note: This is intended for internal use, for example, to initialize a
  ///   tensor attribute from `convolved2D`'s `strides` argument.
  @inlinable
  internal init(_ scalars: (Scalar, Scalar, Scalar, Scalar), on device: Device = .default) {
    self.init([scalars.0, scalars.1, scalars.2, scalars.3], on: device)
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: init(repeating:shape:on:))
  static func _vjpInit(
    repeating repeatedValue: __owned Scalar,
    shape: __owned TensorShape,
    on device: Device
  ) -> (value: Tensor, pullback: (Tensor) -> Scalar) {
    return (
      Tensor(repeating: repeatedValue, shape: shape, on: device),
      {
        $0.sum().scalarized()
      }
    )
  }
}

//===------------------------------------------------------------------------------------------===//
// Casting
//===------------------------------------------------------------------------------------------===//

extension Tensor where Scalar: Numeric {
  /// Perform an element-wise type conversion from a `Bool` tensor.
  @inlinable
  public init(_ other: Tensor<Bool>) {
    self = _Raw.cast(other)
  }

  /// Perform an element-wise conversion from another `Tensor`.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint, OtherScalar: TensorFlowFloatingPoint)
  public init<OtherScalar: Numeric>(_ other: Tensor<OtherScalar>) {
    self = _Raw.cast(other)
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: init(_:))
  static func _vjpCast<OtherScalar: TensorFlowFloatingPoint>(
    _ other: __owned Tensor<OtherScalar>
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor<OtherScalar>) {
    (Tensor(other), { v in Tensor<OtherScalar>(v) })
  }
}

//===------------------------------------------------------------------------------------------===//
// Stacking / Concatenating / Tiling
//===------------------------------------------------------------------------------------------===//

extension Tensor {
  /// Creates a tensor from an array of tensors (which may themselves be scalars).
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public init(_ elements: [Tensor]) {
    self = _Raw.pack(elements)
  }

  /// Stacks `tensors`, along the `axis` dimension, into a new tensor with rank one higher than
  /// the current tensor and each tensor in `tensors`.
  ///
  /// Given that `tensors` all have shape `[A, B, C]`, and `tensors.count = N`, then:
  /// - if `axis == 0` then the resulting tensor will have the shape `[N, A, B, C]`.
  /// - if `axis == 1` then the resulting tensor will have the shape `[A, N, B, C]`.
  /// - etc.
  ///
  /// For example:
  /// ```
  /// // 'x' is [1, 4]
  /// // 'y' is [2, 5]
  /// // 'z' is [3, 6]
  /// Tensor(stacking: [x, y, z]) // is [[1, 4], [2, 5], [3, 6]]
  /// Tensor(stacking: [x, y, z], alongAxis: 1) // is [[1, 2, 3], [4, 5, 6]]
  /// ```
  ///
  /// This is the opposite of `Tensor.unstacked(alongAxis:)`.
  ///
  /// - Parameters:
  ///   - tensors: Tensors to stack.
  ///   - axis: Dimension along which to stack. Negative values wrap around.
  ///
  /// - Precondition: All tensors must have the same shape.
  /// - Precondition: `axis` must be in the range `[-rank, rank)`, where `rank` is the rank of the
  ///   provided tensors.
  ///
  /// - Returns: The stacked tensor.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public init(stacking tensors: [Tensor], alongAxis axis: Int = 0) {
    self = _Raw.pack(tensors, axis: Int64(axis))
  }

  /// Concatenates `tensors` along the `axis` dimension.
  ///
  /// Given that `tensors[i].shape = [D0, D1, ... Daxis(i), ...Dn]`, then the concatenated result
  /// has shape `[D0, D1, ... Raxis, ...Dn]`, where `Raxis = sum(Daxis(i))`. That is, the data
  /// from the input tensors is joined along the `axis` dimension.
  ///
  /// For example:
  /// ```
  /// // t1 is [[1, 2, 3], [4, 5, 6]]
  /// // t2 is [[7, 8, 9], [10, 11, 12]]
  /// Tensor(concatenating: [t1, t2]) // is [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
  /// Tensor(concatenating: [t1, t2], alongAxis: 1) // is [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
  ///
  /// // t3 has shape [2, 3]
  /// // t4 has shape [2, 3]
  /// Tensor(concatenating: [t3, t4]) // has shape [4, 3]
  /// Tensor(concatenating: [t3, t4], alongAxis: 1) // has shape [2, 6]
  /// ```
  ///
  /// - Note: If you are concatenating along a new axis consider using
  ///   `Tensor.init(stacking:alongAxis:)`.
  ///
  /// - Parameters:
  ///   - tensors: Tensors to concatenate.
  ///   - axis: Dimension along which to concatenate. Negative values wrap around.
  ///
  /// - Precondition: All tensors must have the same rank and all dimensions except `axis`
  ///   must be equal.
  /// - Precondition: `axis` must be in the range `[-rank, rank)`, where `rank` is the rank of the
  ///   provided tensors.
  ///
  /// - Returns: The concatenated tensor.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public init(concatenating tensors: [Tensor], alongAxis axis: Int = 0) {
    precondition(tensors.count > 0)
    self = _Raw.concatV2(tensors, axis: Tensor<Int32>(Int32(axis), on: tensors.first!.device))
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: init(_:))
  static func _vjpInitElements(
    _ elements: __owned [Tensor]
  ) -> (value: Tensor, pullback: (Tensor) -> Array<Tensor>.DifferentiableView) {
    _vjpStacking(stacking: elements)
  }

  @inlinable
  @derivative(of: init(stacking:alongAxis:))
  static func _vjpStacking(
    stacking tensors: __owned [Tensor],
    alongAxis axis: __owned Int = 0
  ) -> (value: Tensor, pullback: (Tensor) -> Array<Tensor>.DifferentiableView) {
    (
      Tensor(stacking: tensors, alongAxis: axis),
      { v in
        Array<Tensor>.DifferentiableView(v.unstacked(alongAxis: axis))
      }
    )
  }

  @inlinable
  @derivative(of: init(concatenating:alongAxis:))
  static func _vjpConcatenating(
    concatenating tensors: __owned [Tensor],
    alongAxis axis: __owned Int = 0
  ) -> (value: Tensor, pullback: (Tensor) -> Array<Tensor>.DifferentiableView) {
    let result = Tensor<Scalar>(concatenating: tensors, alongAxis: axis)
    let posAxis = axis < 0 ? axis + tensors[0].rank : axis
    let sizes = Tensor<Int32>(stacking: tensors.map { $0.shapeTensor[posAxis] })
    return (
      result,
      { [count = tensors.count] v in
        if count == 1 { return Array<Tensor>.DifferentiableView([v]) }
        let splits = v.split(sizes: sizes, alongAxis: posAxis)
        return Array<Tensor>.DifferentiableView(splits)
      }
    )
  }
}

//===------------------------------------------------------------------------------------------===//
// Numeric
//===------------------------------------------------------------------------------------------===//

extension Tensor where Scalar: Numeric {
  /// Creates a tensor with all scalars set to zero.
  ///
  /// - Parameter shape: Shape of the tensor.
  @inlinable
  public init(zeros shape: TensorShape, on device: Device = .default) {
    self.init(repeating: 0, shape: shape, on: device)
  }

  /// Creates a tensor with all scalars set to one.
  ///
  /// - Parameter shape: Shape of the tensor.
  @inlinable
  public init(ones shape: TensorShape, on device: Device = .default) {
    self.init(repeating: 1, shape: shape, on: device)
  }

  /// Creates a tensor with all scalars set to zero that has the same shape and type as the provided
  /// tensor.
  ///
  /// - Parameter other: Tensor whose shape and data type to use.
  @inlinable
  public init(zerosLike other: Tensor) {
    self = _Raw.zerosLike(other)
  }

  /// Creates a tensor with all scalars set to one that has the same shape and type as the provided
  /// tensor.
  ///
  /// - Parameter other: Tensor whose shape and data type to use.
  @inlinable
  public init(onesLike other: Tensor) {
    self = _Raw.onesLike(other)
  }

  /// Creates a 1-D tensor representing a sequence from a starting value to, but not including,
  /// an end value, stepping by the specified amount.
  ///
  /// - Parameters:
  ///   - start: The starting value to use for the sequence. If the sequence
  ///     contains any values, the first one is `start`.
  ///   - end: An end value to limit the sequence. `end` is never an element of
  ///     the resulting sequence.
  ///   - stride: The amount to step by with each iteration. `stride` must be
  ///     positive.
  @inlinable
  public init(
    rangeFrom start: Scalar, to end: Scalar, stride: Scalar,
    on device: Device = .default
  ) {
    self = _Raw.range(
      start: Tensor(start, on: device), limit: Tensor(end, on: device),
      delta: Tensor(stride, on: device))
  }

  /// Creates a 1-D tensor representing a sequence from a starting value to, but not including, an
  /// end value, stepping by the specified amount.
  ///
  /// - Parameters:
  ///   - start: The starting value to use for the sequence. If the sequence contains any values,
  ///     the first one is `start`.
  ///   - end: An end value to limit the sequence. `end` is never an element of the resulting
  ///     sequence.
  ///   - stride: The amount to step by with each iteration. `stride` must be positive.
  @inlinable
  public init(rangeFrom start: Tensor<Scalar>, to end: Tensor<Scalar>, stride: Tensor<Scalar>) {
    self = _Raw.range(start: start, limit: end, delta: stride)
  }
}
