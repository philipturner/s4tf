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

extension Tensor/*: ElementaryFunctions*/ where Scalar: TensorFlowFloatingPoint {
  /// The square root of `x`.
  @differentiable(reverse)
  public static func sqrt(_ x: Self) -> Self {
    fatalError()
  }

  @inlinable
  @derivative(of: sqrt)
  internal static func _vjpSqrt(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    fatalError()
  }

}

extension Tensor where Scalar: Numeric {
  /// Adds the scalar to every scalar of the tensor and produces the sum.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public static func + (lhs: Scalar, rhs: Tensor) -> Tensor {
    return Tensor(lhs/*, deviceAndPrecisionLike: rhs*/) + rhs
  }

  /// Adds the scalar to every scalar of the tensor and produces the sum.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public static func + (lhs: Tensor, rhs: Scalar) -> Tensor {
    return lhs + Tensor(rhs/*, deviceAndPrecisionLike: lhs*/)
  }

  /// Subtracts the scalar from every scalar of the tensor and produces the difference.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public static func - (lhs: Scalar, rhs: Tensor) -> Tensor {
    return Tensor(lhs/*, deviceAndPrecisionLike: rhs*/) - rhs
  }

  /// Subtracts the scalar from every scalar of the tensor and produces the difference
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public static func - (lhs: Tensor, rhs: Scalar) -> Tensor {
    return lhs - Tensor(rhs/*, deviceAndPrecisionLike: lhs*/)
  }

  /// Adds two tensors and stores the result in the left-hand-side variable.
  /// - Note: `+=` supports broadcasting.
  @inlinable
  public static func += (lhs: inout Tensor, rhs: Tensor) {
    lhs = lhs + rhs
  }

  /// Adds the scalar to every scalar of the tensor and stores the result in the left-hand-side
  /// variable.
  @inlinable
  public static func += (lhs: inout Tensor, rhs: Scalar) {
    lhs = lhs + rhs
  }

  /// Subtracts the second tensor from the first and stores the result in the left-hand-side
  /// variable.
  /// - Note: `-=` supports broadcasting.
  @inlinable
  public static func -= (lhs: inout Tensor, rhs: Tensor) {
    lhs = lhs - rhs
  }

  /// Subtracts the scalar from every scalar of the tensor and stores the result in the
  /// left-hand-side variable.
  @inlinable
  public static func -= (lhs: inout Tensor, rhs: Scalar) {
    lhs = lhs - rhs
  }

  /// Returns the tensor produced by multiplying the two tensors.
  /// - Note: `*` supports broadcasting.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public static func * (lhs: Tensor, rhs: Tensor) -> Tensor {
    fatalError()
  }

  /// Returns the tensor by multiplying it with every scalar of the tensor.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public static func * (lhs: Scalar, rhs: Tensor) -> Tensor {
    return Tensor(lhs, deviceAndPrecisionLike: rhs) * rhs
  }

  /// Multiplies the scalar with every scalar of the tensor and produces the product.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public static func * (lhs: Tensor, rhs: Scalar) -> Tensor {
    return lhs * Tensor(rhs, deviceAndPrecisionLike: lhs)
  }

  /// Multiplies two tensors and stores the result in the left-hand-side variable.
  /// - Note: `*=` supports broadcasting.
  @inlinable
  public static func *= (lhs: inout Tensor, rhs: Tensor) {
    lhs = lhs * rhs
  }

  /// Multiplies the tensor with the scalar, broadcasting the scalar, and stores the result in the
  /// left-hand-side variable.
  @inlinable
  public static func *= (lhs: inout Tensor, rhs: Scalar) {
    lhs = lhs * rhs
  }

}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: +)
  static func _vjpAdd(lhs: Tensor, rhs: Scalar) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Scalar)
  ) {
    fatalError()
//    return (lhs + rhs, { v in (v, v.sum().scalarized()) })
  }

  @inlinable
  @derivative(of: +)
  static func _vjpAdd(lhs: Scalar, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Scalar, Tensor)
  ) {
    fatalError()
//    return (lhs + rhs, { v in (v.sum().scalarized(), v) })
  }

  @inlinable
  @derivative(of: -)
  static func _vjpSubtract(lhs: Tensor, rhs: Scalar) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Scalar)
  ) {
    fatalError()
//    return (lhs - rhs, { v in (v, -v.sum().scalarized()) })
  }

  @inlinable
  @derivative(of: -)
  static func _vjpSubtract(lhs: Scalar, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Scalar, Tensor)
  ) {
    fatalError()
//    return (lhs - rhs, { v in (v.sum().scalarized(), -v) })
  }

  @inlinable
  @derivative(of: *)
  static func _vjpMultiply(lhs: Tensor, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    fatalError()

  }

  @inlinable
  @derivative(of: *)
  static func _vjpMultiply(lhs: Tensor, rhs: Scalar) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Scalar)
  ) {
    fatalError()
//    return (lhs * rhs, { v in (v * rhs, (v * lhs).sum().scalarized()) })
  }

  @inlinable
  @derivative(of: *)
  static func _vjpMultiply(lhs: Scalar, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Scalar, Tensor)
  ) {
    fatalError()
//    return (lhs * rhs, { v in ((v * rhs).sum().scalarized(), v * lhs) })
  }



}


extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: +, wrt: lhs)
  static func _vjpAdd(lhs: Tensor, rhs: Scalar) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    return (lhs + rhs, { v in v })
  }

  @inlinable
  @derivative(of: +, wrt: rhs)
  static func _vjpAdd(lhs: Scalar, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    return (lhs + rhs, { v in v })
  }

  @inlinable
  @derivative(of: -, wrt: lhs)
  static func _vjpSubtract(lhs: Tensor, rhs: Scalar) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    return (lhs - rhs, { v in v })
  }

  @inlinable
  @derivative(of: -, wrt: rhs)
  static func _vjpSubtract(lhs: Scalar, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    return (lhs - rhs, { v in -v })
  }

  @inlinable
  @derivative(of: *, wrt: lhs)
  static func _vjpMultiply(lhs: Tensor, rhs: Scalar) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    return (lhs * rhs, { v in v * rhs })
  }

  @inlinable
  @derivative(of: *, wrt: rhs)
  static func _vjpMultiply(lhs: Scalar, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    return (lhs * rhs, { v in v * lhs })
  }

//  }
}

//
extension Tensor where Scalar: SignedNumeric {
  /// Returns the negation of the specified tensor element-wise.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public static prefix func - (rhs: Tensor) -> Tensor {
    fatalError()
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: -)
  static func _vjpNegate(_ x: Tensor) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    return (-x, { v in -v })
  }
}

/// Returns the square of the tensor.
extension Tensor where Scalar: Numeric {
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func squared() -> Tensor {
    fatalError()
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: squared)
  func _vjpSquared() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (squared(), { 2 * self * $0 })
  }
}

/// Returns the square root of the specified tensor element-wise.
@inlinable
@differentiable(reverse)
public func sqrt<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.sqrt(x)
}

/// Returns the inverse square root of the specified tensor element-wise.
@inlinable
@differentiable(reverse)
public func rsqrt<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  fatalError()
}

@inlinable
@derivative(of: rsqrt)
internal func _vjpRsqrt<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> Tensor<T>) {
  fatalError()
}


public struct Moments<Scalar: TensorFlowFloatingPoint>: Differentiable {
  public var mean: Tensor<Scalar>
  public var variance: Tensor<Scalar>

  @differentiable(reverse)
  public init(mean: Tensor<Scalar>, variance: Tensor<Scalar>) {
    self.mean = mean
    self.variance = variance
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  /// Returns the mean and variance of this tensor along the specified axes. The reduced
  /// dimensions are removed.
  ///
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: `axes` must have rank `1`.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(reverse, wrt: self)
  public func moments(squeezingAxes axes: Tensor<Int32>) -> Moments<Scalar> {
//    ensureValid(axes: axes)
//    let mean = self.mean(alongAxes: axes)
//    let variance = squaredDifference(self, self)//.mean(squeezingAxes: axes)
    return Moments(
      // The following is required because `Tensor.squeezingShape(at:)` does not accept
      // `Tensor<Int32>`-valued arguments.
      mean: self,//.sum(squeezingAxes: axes),
      variance: self)
  }

  /// Returns the mean and variance of this tensor along the specified axes. The reduced
  /// dimensions are removed.
  ///
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(reverse, wrt: self)
  public func moments(squeezingAxes axes: [Int]) -> Moments<Scalar> {
//    ensureValid(axes: axes)
//    let mean = self.mean(squeezingAxes: axes)
//    let variance = squaredDifference(self, self)//.mean(squeezingAxes: axes)
    return Moments(mean: self, variance: self)
  }

  /// Returns the mean and variance of this tensor along the specified axes. The reduced
  /// dimensions are removed.
  ///
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(reverse, wrt: self)
  public func moments(squeezingAxes axes: Int...) -> Moments<Scalar> {
    moments(squeezingAxes: axes)
  }

  /// Returns the mean and variance of this tensor's elements.
  @inlinable
  @differentiable(reverse, wrt: self)
  public func moments() -> Moments<Scalar> {
    moments(squeezingAxes: Array(0..<shape.rank))
  }

  /// Returns the mean and variance of this tensor along the specified axes. The reduced
  /// dimensions are retained with value `1`.
  ///
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: `axes` must have rank `1`.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(reverse, wrt: self)
  public func moments(alongAxes axes: Tensor<Int32>) -> Moments<Scalar> {
//    ensureValid(axes: axes)
//    let mean = self.mean(alongAxes: axes)
//    let variance = squaredDifference(self, self)//.mean(alongAxes: axes)
    return Moments<Scalar>(mean: self, variance: self)
  }

  /// Returns the mean and variance of this tensor along the specified axes. The reduced
  /// dimensions are retained with value `1`.
  ///
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(reverse, wrt: self)
  public func moments(alongAxes axes: [Int]) -> Moments<Scalar> {
//    ensureValid(axes: axes)
//    let mean = self.mean(alongAxes: axes)
//    let variance = squaredDifference(self, self)//.mean(alongAxes: axes)
    return Moments<Scalar>(mean: self, variance: self)
  }

  /// Returns the mean and variance of this tensor along the specified axes. The reduced
  /// dimensions are retained with value `1`.
  ///
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(reverse, wrt: self)
  public func moments(alongAxes axes: Int...) -> Moments<Scalar> {
    moments(alongAxes: axes)
  }
}

//===------------------------------------------------------------------------------------------===//
