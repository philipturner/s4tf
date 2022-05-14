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
#if TENSORFLOW_USE_STANDARD_TOOLCHAIN
import Numerics
#endif

infix operator .>: ComparisonPrecedence
infix operator .==: ComparisonPrecedence

// TODO:
// - Consider explicit broadcasting for elementwise binary ops when
//   scalarization and rank getter are implemented.

// TODO: Remove the following extension once `./` and `./=` are defined for
// `PointwiseMultiplicative`.

infix operator ./: MultiplicationPrecedence
infix operator ./=: AssignmentPrecedence

extension PointwiseMultiplicative {
  public static func ./ (lhs: Self, rhs: Self) -> Self {
    lhs .* rhs.reciprocal
  }

  public static func ./= (lhs: inout Self, rhs: Self) {
    lhs = lhs ./ rhs
  }
}

//===------------------------------------------------------------------------------------------===//
// Generic Elementary Functions
//===------------------------------------------------------------------------------------------===//

extension Tensor: ElementaryFunctions where Scalar: TensorFlowFloatingPoint {
  /// The square root of `x`.
  ///
  /// For real types, if `x` is negative the result is `.nan`. For complex
  /// types there is a branch cut on the negative real axis.
  @differentiable(reverse)
  public static func sqrt(_ x: Self) -> Self {
    _Raw.sqrt(x)
  }

  @inlinable
  @derivative(of: sqrt)
  internal static func _vjpSqrt(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let value = Tensor.sqrt(x)
    return (value, { v in v / (2 * value) })
  }

  /// The cosine of `x`, interpreted as an angle in radians.
  @differentiable(reverse)
  public static func cos(_ x: Self) -> Self {
    _Raw.cos(x)
  }

  @inlinable
  @derivative(of: cos)
  internal static func _vjpCos(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (cos(x), { v in -v * sin(x) })
  }

  /// The sine of `x`, interpreted as an angle in radians.
  @differentiable(reverse)
  public static func sin(_ x: Self) -> Self {
    _Raw.sin(x)
  }

  @inlinable
  @derivative(of: sin)
  internal static func _vjpSin(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (sin(x), { v in v * cos(x) })
  }

  /// The tangent of `x`, interpreted as an angle in radians.
  @differentiable(reverse)
  public static func tan(_ x: Self) -> Self {
    _Raw.tan(x)
  }

  @inlinable
  @derivative(of: tan)
  internal static func _vjpTan(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let value = tan(x)
    return (value, { v in v * (1 + value.squared()) })
  }

  /// The inverse cosine of `x` in radians.
  @differentiable(reverse)
  public static func acos(_ x: Self) -> Self {
    _Raw.acos(x)
  }

  @inlinable
  @derivative(of: acos)
  internal static func _vjpAcos(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (acos(x), { v in -v / sqrt(1 - x.squared()) })
  }

  /// The inverse sine of `x` in radians.
  @differentiable(reverse)
  public static func asin(_ x: Self) -> Self {
    _Raw.asin(x)
  }

  @inlinable
  @derivative(of: asin)
  internal static func _vjpAsin(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (asin(x), { v in v / sqrt(1 - x.squared()) })
  }

  /// The inverse tangent of `x` in radians.
  @differentiable(reverse)
  public static func atan(_ x: Self) -> Self {
    _Raw.atan(x)
  }

  @inlinable
  @derivative(of: atan)
  internal static func _vjpAtan(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (atan(x), { v in v / (1 + x.squared()) })
  }

  /// The hyperbolic cosine of `x`.
  @differentiable(reverse)
  public static func cosh(_ x: Self) -> Self {
    _Raw.cosh(x)
  }

  @inlinable
  @derivative(of: cosh)
  internal static func _vjpCosh(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (cosh(x), { v in v * sinh(x) })
  }

  /// The hyperbolic sine of `x`.
  @differentiable(reverse)
  public static func sinh(_ x: Self) -> Self {
    _Raw.sinh(x)
  }

  @inlinable
  @derivative(of: sinh)
  internal static func _vjpSinh(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (sinh(x), { v in v * cosh(x) })
  }

  /// The hyperbolic tangent of `x`.
  @differentiable(reverse)
  public static func tanh(_ x: Self) -> Self {
    _Raw.tanh(x)
  }

  @inlinable
  @derivative(of: tanh)
  internal static func _vjpTanh(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let value = tanh(x)
    return (value, { v in v * (1 - value.squared()) })
  }

  /// The inverse hyperbolic cosine of `x`.
  @differentiable(reverse)
  public static func acosh(_ x: Self) -> Self {
    _Raw.acosh(x)
  }

  @inlinable
  @derivative(of: acosh)
  internal static func _vjpAcosh(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (acosh(x), { v in v / asinh(x) })
  }

  /// The inverse hyperbolic sine of `x`.
  @differentiable(reverse)
  public static func asinh(_ x: Self) -> Self {
    _Raw.asinh(x)
  }

  @inlinable
  @derivative(of: asinh)
  internal static func _vjpAsinh(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (asinh(x), { v in v / acosh(x) })
  }

  /// The inverse hyperbolic tangent of `x`.
  @differentiable(reverse)
  public static func atanh(_ x: Self) -> Self {
    _Raw.atanh(x)
  }

  @inlinable
  @derivative(of: atanh)
  internal static func _vjpAtanh(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (atanh(x), { v in v / (1 - x.squared()) })
  }

  /// The exponential function applied to `x`, or `e**x`.
  @differentiable(reverse)
  public static func exp(_ x: Self) -> Self {
    _Raw.exp(x)
  }

  @inlinable
  @derivative(of: exp)
  internal static func _vjpExp(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let value = exp(x)
    return (value, { v in value * v })
  }

  /// Two raised to to power `x`.
  @differentiable(reverse)
  public static func exp2(_ x: Self) -> Self {
    pow(Tensor(2, on: x.device), x)
  }

  /// Ten raised to to power `x`.
  @differentiable(reverse)
  public static func exp10(_ x: Self) -> Self {
    pow(Tensor(10, on: x.device), x)
  }

  /// `exp(x) - 1` evaluated so as to preserve accuracy close to zero.
  @differentiable(reverse)
  public static func expm1(_ x: Self) -> Self {
    _Raw.expm1(x)
  }

#if TENSORFLOW_USE_STANDARD_TOOLCHAIN
  @differentiable(reverse)
  public static func expMinusOne(_ x: Self) -> Self {
    return expm1(x)
  }
#endif

  @inlinable
  @derivative(of: expm1)
  internal static func _vjpExpm1(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let y = expm1(x)
    return (y, { v in v * y })
  }

  /// The natural logarithm of `x`.
  @differentiable(reverse)
  public static func log(_ x: Self) -> Self {
    _Raw.log(x)
  }

  @inlinable
  @derivative(of: log(_:))
  internal static func _vjpLog(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (log(x), { v in v / x })
  }

  /// The base-two logarithm of `x`.
  @differentiable(reverse)
  public static func log2(_ x: Self) -> Self {
    log(x) / Scalar.log(2)
  }

  /// The base-ten logarithm of `x`.
  @differentiable(reverse)
  public static func log10(_ x: Self) -> Self {
    log(x) / Scalar.log(10)
  }

  /// `log(1 + x)` evaluated so as to preserve accuracy close to zero.
  @differentiable(reverse)
  public static func log1p(_ x: Self) -> Self {
    _Raw.log1p(x)
  }

#if TENSORFLOW_USE_STANDARD_TOOLCHAIN
  @differentiable(reverse)
  public static func log(onePlus x: Self) -> Self {
    return log1p(x)
  }
#endif

  @inlinable
  @derivative(of: log1p)
  internal static func _vjpLog1p(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (log1p(x), { v in _Raw.xdivy(v, 1 + x) })
  }

  /// `exp(y log(x))` computed without loss of intermediate precision.
  ///
  /// For real types, if `x` is negative the result is NaN, even if `y` has
  /// an integral value. For complex types, there is a branch cut on the
  /// negative real axis.
  @differentiable(reverse)
  public static func pow(_ x: Self, _ y: Self) -> Self {
    _Raw.pow(x, y)
  }

  @inlinable
  @derivative(of: pow)
  internal static func _vjpPow(
    _ x: Tensor, _ y: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
    fatalError()
  }

  /// `x` raised to the `n`th power.
  ///
  /// The product of `n` copies of `x`.
  @differentiable(reverse)
  public static func pow(_ x: Self, _ n: Int) -> Self {
    pow(x, Tensor(Scalar(n), on: x.device))
  }

  public static func root(_ x: Self, _ n: Int) -> Self {
    fatalError()
  }
}

//===------------------------------------------------------------------------------------------===//
// Vector Space
//===------------------------------------------------------------------------------------------===//

extension Tensor: VectorProtocol where Scalar: TensorFlowFloatingPoint {
  public typealias VectorSpaceScalar = Float

  // @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public func scaled(by scale: Float) -> Self {
    Scalar(scale) * self
  }

  // @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public func adding(_ scalar: Float) -> Self {
    self + Scalar(scalar)
  }

  // @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public func subtracting(_ scalar: Float) -> Self {
    self - Scalar(scalar)
  }
}

//===------------------------------------------------------------------------------------------===//
// Additional Element-wise Operators
//===------------------------------------------------------------------------------------------===//

extension Tensor where Scalar: Numeric {
  /// Adds the scalar to every scalar of the tensor and produces the sum.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public static func + (lhs: Scalar, rhs: Tensor) -> Tensor {
    return Tensor(lhs, deviceAndPrecisionLike: rhs) + rhs
  }

  /// Adds the scalar to every scalar of the tensor and produces the sum.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public static func + (lhs: Tensor, rhs: Scalar) -> Tensor {
    return lhs + Tensor(rhs, deviceAndPrecisionLike: lhs)
  }

  /// Subtracts the scalar from every scalar of the tensor and produces the difference.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public static func - (lhs: Scalar, rhs: Tensor) -> Tensor {
    return Tensor(lhs, deviceAndPrecisionLike: rhs) - rhs
  }

  /// Subtracts the scalar from every scalar of the tensor and produces the difference
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public static func - (lhs: Tensor, rhs: Scalar) -> Tensor {
    return lhs - Tensor(rhs, deviceAndPrecisionLike: lhs)
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
    return _Raw.mul(lhs, rhs)
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

  /// Returns the quotient of dividing the first tensor by the second.
  /// - Note: `/` supports broadcasting.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public static func / (lhs: Tensor, rhs: Tensor) -> Tensor {
    return _Raw.div(lhs, rhs)
  }

  /// Returns the quotient of dividing the scalar by the tensor, broadcasting the scalar.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public static func / (lhs: Scalar, rhs: Tensor) -> Tensor {
    return Tensor(lhs, deviceAndPrecisionLike: rhs) / rhs
  }

  /// Returns the quotient of dividing the tensor by the scalar, broadcasting the scalar.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public static func / (lhs: Tensor, rhs: Scalar) -> Tensor {
    return lhs / Tensor(rhs, deviceAndPrecisionLike: lhs)
  }

  /// Divides the first tensor by the second and stores the quotient in the left-hand-side
  /// variable.
  @inlinable
  public static func /= (lhs: inout Tensor, rhs: Tensor) {
    lhs = lhs / rhs
  }

  /// Divides the tensor by the scalar, broadcasting the scalar, and stores the quotient in the
  /// left-hand-side variable.
  @inlinable
  public static func /= (lhs: inout Tensor, rhs: Scalar) {
    lhs = lhs / rhs
  }

  /// Returns the remainder of dividing the first tensor by the second.
  /// - Note: `%` supports broadcasting.
  @inlinable
  public static func % (lhs: Tensor, rhs: Tensor) -> Tensor {
    return _Raw.mod(lhs, rhs)
  }

  /// Returns the remainder of dividing the tensor by the scalar, broadcasting the scalar.
  @inlinable
  public static func % (lhs: Tensor, rhs: Scalar) -> Tensor {
    return lhs % Tensor(rhs, deviceAndPrecisionLike: lhs)
  }

  /// Returns the remainder of dividing the scalar by the tensor, broadcasting the scalar.
  @inlinable
  public static func % (lhs: Scalar, rhs: Tensor) -> Tensor {
    return Tensor(lhs, deviceAndPrecisionLike: rhs) % rhs
  }

  /// Divides the first tensor by the second and stores the remainder in the left-hand-side
  /// variable.
  @inlinable
  public static func %= (lhs: inout Tensor, rhs: Tensor) {
    lhs = lhs % rhs
  }

  /// Divides the tensor by the scalar and stores the remainder in the left-hand-side variable.
  @inlinable
  public static func %= (lhs: inout Tensor, rhs: Scalar) {
    lhs = lhs % rhs
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: +)
  static func _vjpAdd(lhs: Tensor, rhs: Scalar) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Scalar)
  ) {
    return (lhs + rhs, { v in (v, v.sum().scalarized()) })
  }

  @inlinable
  @derivative(of: +)
  static func _vjpAdd(lhs: Scalar, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Scalar, Tensor)
  ) {
    return (lhs + rhs, { v in (v.sum().scalarized(), v) })
  }

  @inlinable
  @derivative(of: -)
  static func _vjpSubtract(lhs: Tensor, rhs: Scalar) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Scalar)
  ) {
    return (lhs - rhs, { v in (v, -v.sum().scalarized()) })
  }

  @inlinable
  @derivative(of: -)
  static func _vjpSubtract(lhs: Scalar, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Scalar, Tensor)
  ) {
    return (lhs - rhs, { v in (v.sum().scalarized(), -v) })
  }

  @inlinable
  @derivative(of: *)
  static func _vjpMultiply(lhs: Tensor, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    return (
      lhs * rhs,
      { [broadcastPb = BroadcastingPullback(lhs, rhs)] v in
        return broadcastPb(rhs * v, lhs * v)
      }
    )
  }

  @inlinable
  @derivative(of: *)
  static func _vjpMultiply(lhs: Tensor, rhs: Scalar) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Scalar)
  ) {
    return (lhs * rhs, { v in (v * rhs, (v * lhs).sum().scalarized()) })
  }

  @inlinable
  @derivative(of: *)
  static func _vjpMultiply(lhs: Scalar, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Scalar, Tensor)
  ) {
    return (lhs * rhs, { v in ((v * rhs).sum().scalarized(), v * lhs) })
  }

  @inlinable
  @derivative(of: /)
  static func _vjpDivide(lhs: Tensor, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    return (
      lhs / rhs,
      { [broadcastPb = BroadcastingPullback(lhs, rhs)] v in
        return broadcastPb(v / rhs, -lhs / rhs.squared() * v)
      }
    )
  }

  @inlinable
  @derivative(of: /)
  static func _vjpDivide(lhs: Tensor, rhs: Scalar) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Scalar)
  ) {
    return (
      lhs / rhs,
      { v in
        (
          v / rhs,
          (v * -lhs / Tensor(rhs, deviceAndPrecisionLike: lhs).squared()).sum().scalarized()
        )
      }
    )
  }

  @inlinable
  @derivative(of: /)
  static func _vjpDivide(lhs: Scalar, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Scalar, Tensor)
  ) {
    return (lhs / rhs, { v in ((v / rhs).sum().scalarized(), v * -lhs / rhs.squared()) })
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

  @inlinable
  @derivative(of: /, wrt: lhs)
  static func _vjpDivide(lhs: Tensor, rhs: Scalar) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    return (lhs / rhs, { v in v / rhs })
  }

  @inlinable
  @derivative(of: /, wrt: rhs)
  static func _vjpDivide(lhs: Scalar, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    return (lhs / rhs, { v in v * -lhs / rhs.squared() })
  }
}

extension Tensor where Scalar == Bool {
  /// Returns `!self` element-wise.
  @inlinable
  public func elementsLogicalNot() -> Tensor {
    return _Raw.logicalNot(self)
  }

  /// Returns `self && other` element-wise.
  /// - Note: `&&` supports broadcasting.
  @inlinable
  public func elementsLogicalAnd(_ other: Tensor) -> Tensor {
    return _Raw.logicalAnd(self, other)
  }

  /// Returns `self && other` element-wise, broadcasting `other`.
  @inlinable
  public func elementsLogicalAnd(_ other: Scalar) -> Tensor {
    return elementsLogicalAnd(Tensor(other, on: device))
  }

  /// Returns `self || other` element-wise.
  @inlinable
  public func elementsLogicalOr(_ other: Tensor) -> Tensor {
    return _Raw.logicalOr(self, other)
  }

  /// Returns `self || other` element-wise, broadcasting `other`.
  @inlinable
  public func elementsLogicalOr(_ other: Scalar) -> Tensor {
    return elementsLogicalOr(Tensor(other, on: device))
  }
}

extension Tensor where Scalar: TensorFlowNumeric {
  /// Returns `max(min(self, max), min)`.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public func clipped(min: Tensor, max: Tensor) -> Tensor {
    _Raw.clipByValue(t: self, clipValueMin: min, clipValueMax: max)
  }

  /// Returns `max(min(self, max), min)`.
  @inlinable
  @differentiable(reverse, wrt: (self, min) where Scalar: TensorFlowFloatingPoint)
  public func clipped(min: Tensor, max: Scalar) -> Tensor {
    clipped(min: min, max: Tensor(max, deviceAndPrecisionLike: self))
  }

  /// Returns `max(min(self, max), min)`.
  @inlinable
  @differentiable(reverse, wrt: (self, max) where Scalar: TensorFlowFloatingPoint)
  public func clipped(min: Scalar, max: Tensor) -> Tensor {
    clipped(min: Tensor(min, deviceAndPrecisionLike: self), max: max)
  }

  /// Returns `max(min(self, max), min)`.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func clipped(min: Scalar, max: Scalar) -> Tensor {
    clipped(
      min: Tensor(min, deviceAndPrecisionLike: self),
      max: Tensor(max, deviceAndPrecisionLike: self))
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: clipped)
  func _vjpClipped(min: Tensor, max: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor, Tensor)
  ) {
    fatalError()
  }
}

//===------------------------------------------------------------------------------------------===//
// Element-wise Unary Math Functions
//===------------------------------------------------------------------------------------------===//

// Export Glibc/Darwin/ucrt math functions. We should not require users to import
// Foundation/Darwin/Glibc/ucrt in order to use scalar math functions.

#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
  @_exported import Darwin.C
#elseif os(Windows)
  @_exported import ucrt
#else
  @_exported import Glibc
#endif


extension Tensor where Scalar: SignedNumeric {
  /// Returns the negation of the specified tensor element-wise.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public static prefix func - (rhs: Tensor) -> Tensor {
    return _Raw.neg(rhs)
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: -)
  static func _vjpNegate(_ x: Tensor) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    return (-x, { v in -v })
  }
}

/// Returns the absolute value of the specified tensor element-wise.
@inlinable
@differentiable(reverse where T: TensorFlowFloatingPoint)
public func abs<T: SignedNumeric>(_ x: Tensor<T>) -> Tensor<T> {
  _Raw.abs(x)
}

@inlinable
@derivative(of: abs)
internal func _vjpAbs<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> Tensor<T>) {
  let sign = _Raw.sign(x)
  return (abs(x), { v in v * sign })
}

/// Returns the natural logarithm of the specified tensor element-wise.
@inlinable
@differentiable(reverse)
public func log<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.log(x)
}

/// Returns the base-two logarithm of the specified tensor element-wise.
@inlinable
@differentiable(reverse)
public func log2<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  log(x) / T.log(2)
}

/// Returns the base-ten logarithm of the specified tensor element-wise.
@inlinable
@differentiable(reverse)
public func log10<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  log(x) / T.log(10)
}

/// Returns the logarithm of `1 + x` element-wise.
@inlinable
@differentiable(reverse)
public func log1p<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.log1p(x)
}

/// Returns `log(1 - exp(x))` using a numerically stable approach.
///
/// - Note: The approach is shown in Equation 7 of:
///   https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf.
@inlinable
public func log1mexp<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  fatalError()
}

/// Returns the sine of the specified tensor element-wise.
@inlinable
@differentiable(reverse)
public func sin<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.sin(x)
}

/// Returns the cosine of the specified tensor element-wise.
@inlinable
@differentiable(reverse)
public func cos<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.cos(x)
}

/// Returns the tangent of the specified tensor element-wise.
@inlinable
@differentiable(reverse)
public func tan<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.tan(x)
}

/// Returns the hyperbolic sine of the specified tensor element-wise.
@inlinable
@differentiable(reverse)
public func sinh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.sinh(x)
}

/// Returns the hyperbolic cosine of the specified tensor element-wise.
@inlinable
@differentiable(reverse)
public func cosh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.cosh(x)
}

/// Returns the hyperbolic tangent of the specified tensor element-wise.
@inlinable
@differentiable(reverse)
public func tanh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.tanh(x)
}

/// Returns the inverse cosine of the specified tensor element-wise.
@inlinable
@differentiable(reverse)
public func acos<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.acos(x)
}

/// Returns the inverse sine of the specified tensor element-wise.
@inlinable
@differentiable(reverse)
public func asin<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.asin(x)
}

/// Returns the inverse tangent of the specified tensor element-wise.
@inlinable
@differentiable(reverse)
public func atan<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.atan(x)
}

/// Returns the inverse hyperbolic cosine of the specified tensor element-wise.
@inlinable
@differentiable(reverse)
public func acosh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.acosh(x)
}

/// Returns the inverse hyperbolic sine of the specified tensor element-wise.
@inlinable
@differentiable(reverse)
public func asinh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.asinh(x)
}

/// Returns the inverse hyperbolic tangent of the specified tensor element-wise.
@inlinable
@differentiable(reverse)
public func atanh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.atanh(x)
}

/// Returns the square of the tensor.
extension Tensor where Scalar: Numeric {
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func squared() -> Tensor {
    _Raw.square(self)
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
  _Raw.rsqrt(x)
}

@inlinable
@derivative(of: rsqrt)
internal func _vjpRsqrt<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> Tensor<T>) {
  let value = rsqrt(x)
  return (value, { v in _Raw.rsqrtGrad(value, dy: v) })
}

/// Returns the exponential of the specified tensor element-wise.
@inlinable
@differentiable(reverse)
public func exp<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Tensor.exp(x)
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  /// Returns a boolean tensor indicating which elements of `x` are finite.
  @inlinable public var isFinite: Tensor<Bool> { _Raw.isFinite(self) }

  /// Returns a boolean tensor indicating which elements of `x` are infinite.
  @inlinable public var isInfinite: Tensor<Bool> { _Raw.isInf(self) }

  /// Returns a boolean tensor indicating which elements of `x` are NaN-valued.
  @inlinable public var isNaN: Tensor<Bool> { _Raw.isNan(self) }
}

//===------------------------------------------------------------------------------------------===//
// Element-wise Binary Math Functions
//===------------------------------------------------------------------------------------------===//

/// Returns the squared difference between `x` and `y`.
///// - Returns: `(x - y) ^ 2`.
//@inlinable
//@differentiable(reverse where T: TensorFlowFloatingPoint)
//public func squaredDifference<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T> {
//  _Raw.squaredDifference(x, y)
//}
//
//@inlinable
//@derivative(of: squaredDifference)
//internal func _vjpSquaredDifference<T: TensorFlowFloatingPoint>(
//  _ x: Tensor<T>,
//  _ y: Tensor<T>
//) -> (value: Tensor<T>, pullback: (Tensor<T>) -> (Tensor<T>, Tensor<T>)) {
//  (
//    squaredDifference(x, y),
//    { seed in
//      let lhsGrad = 2 * seed * (x - y)
//      return BroadcastingPullback(x, y)(lhsGrad, -lhsGrad)
//    }
//  )
//}

/// Returns the element-wise maximum of two tensors.
/// - Note: `max` supports broadcasting.
@inlinable
@differentiable(reverse where T: TensorFlowFloatingPoint)
public func max<T>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> where T: Numeric & Comparable {
  _Raw.maximum(lhs, rhs)
}

@inlinable
@derivative(of: max)
internal func _vjpMax<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> (Tensor<T>, Tensor<T>)) {
  fatalError()
}

/// Returns the element-wise maximum of the scalar and the tensor, broadcasting the scalar.
@inlinable
@differentiable(reverse, wrt: rhs where T: TensorFlowFloatingPoint)
public func max<T>(_ lhs: T, _ rhs: Tensor<T>) -> Tensor<T> where T: Numeric & Comparable {
  max(Tensor(lhs, deviceAndPrecisionLike: rhs), rhs)
}

/// Returns the element-wise maximum of the scalar and the tensor, broadcasting the scalar.
@inlinable
@differentiable(reverse, wrt: lhs where T: TensorFlowFloatingPoint)
public func max<T>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<T> where T: Numeric & Comparable {
  max(lhs, Tensor(rhs, deviceAndPrecisionLike: lhs))
}

/// Returns the element-wise minimum of two tensors.
/// - Note: `min` supports broadcasting.
@inlinable
@differentiable(reverse where T: TensorFlowFloatingPoint)
public func min<T>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> where T: Numeric & Comparable {
  _Raw.minimum(lhs, rhs)
}

@inlinable
@derivative(of: min)
internal func _vjpMin<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> (value: Tensor<T>, pullback: (Tensor<T>) -> (Tensor<T>, Tensor<T>)) {
  fatalError()
}

/// Returns the element-wise minimum of the scalar and the tensor, broadcasting the scalar.
@inlinable
@differentiable(reverse, wrt: rhs where T: TensorFlowFloatingPoint)
public func min<T>(_ lhs: T, _ rhs: Tensor<T>) -> Tensor<T> where T: Numeric & Comparable {
  min(Tensor(lhs, deviceAndPrecisionLike: rhs), rhs)
}

/// Returns the element-wise minimum of the scalar and the tensor, broadcasting the scalar.
@inlinable
@differentiable(reverse, wrt: lhs where T: TensorFlowFloatingPoint)
public func min<T>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<T> where T: Numeric & Comparable {
  min(lhs, Tensor(rhs, deviceAndPrecisionLike: lhs))
}

// Note: adapted from `_MinOrMaxGrad`:
// https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/math_grad.py#L223.
@inlinable
internal func _vjpMinMaxHelper<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>,
  _ y: Tensor<T>,
  originalValue: Tensor<T>,
  seed: Tensor<T>,
  comparisonOperation: (Tensor<T>, Tensor<T>) -> Tensor<Bool>
) -> (value: Tensor<T>, pullback: Tensor<T>) {
  let mask = Tensor<T>(comparisonOperation(x, y))
  let lhsGrad = seed * mask
  let rhsGrad = seed * (1 - mask)
  let (lhsShape, rhsShape) = (x.shapeTensor, y.shapeTensor)
  let (lhsAxes, rhsAxes) = _Raw.broadcastGradientArgs(s0: lhsShape, s1: rhsShape)
  return (
    lhsGrad.sum(squeezingAxes: lhsAxes).reshaped(toShape: lhsShape),
    rhsGrad.sum(squeezingAxes: rhsAxes).reshaped(toShape: rhsShape)
  )
}

/// Returns the cosine similarity between `x` and `y`.
@differentiable(reverse)
public func cosineSimilarity<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  _ y: Tensor<Scalar>
) -> Tensor<Scalar> {
  (x * y).sum() / (sqrt(x.squared().sum()) * sqrt(y.squared().sum()))
}

/// Returns the cosine distance between `x` and `y`. Cosine distance is defined as
/// `1 - cosineSimilarity(x, y)`.
@differentiable(reverse)
public func cosineDistance<Scalar: TensorFlowFloatingPoint>(
  _ x: Tensor<Scalar>,
  _ y: Tensor<Scalar>
) -> Tensor<Scalar> {
  1 - cosineSimilarity(x, y)
}

//===------------------------------------------------------------------------------------------===//
// Selection Functions
//===------------------------------------------------------------------------------------------===//

extension Tensor {
  /// Replaces elements of this tensor with `other` in the lanes where `mask` is
  /// `true`.
  ///
  /// - Precondition: `self` and `other` must have the same shape. If
  ///   `self` and `other` are scalar, then `mask` must also be scalar. If
  ///   `self` and `other` have rank greater than or equal to `1`, then `mask`
  ///   must be either have the same shape as `self` or be a 1-D `Tensor` such
  ///   that `mask.scalarCount == self.shape[0]`.
  @inlinable
  @differentiable(reverse, wrt: (self, other) where Scalar: TensorFlowFloatingPoint)
  public func replacing(with other: Tensor, where mask: Tensor<Bool>) -> Tensor {
    precondition(self.shape == other.shape, "`self` and `other` must have the same shape.")
    return _Raw.select(condition: mask, t: other, e: self)
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: replacing)
  func _vjpReplacing(
    with other: Tensor,
    where mask: Tensor<Bool>
  ) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
    return (
      replacing(with: other, where: mask),
      { v in
        let zeros = Tensor(zerosLike: v)
        return (v.replacing(with: zeros, where: mask), zeros.replacing(with: v, where: mask))
      }
    )
  }
}

//===------------------------------------------------------------------------------------------===//
// Reduction Functions
//===------------------------------------------------------------------------------------------===//

extension Tensor where Scalar == Bool {
  /// Returns `true` if all scalars are equal to `true`. Otherwise, returns `false`.
  // NOTE: This overload is necessary, otherwise `all()` would refer to the variadic method
  // `all(squeezingAxes:)` with zero indices.
  @inlinable
  public func all() -> Bool {
    let axes = Tensor<Int32>(rangeFrom: 0, to: Int32(rank), stride: 1, on: device)
    return _Raw.all(self, reductionIndices: axes).scalarized()
  }

  /// Returns `true` if any scalars are equal to `true`. Otherwise, returns `false`.
  // NOTE: This overload is necessary, otherwise `any()` would refer to the variadic method
  // `any(squeezingAxes:)` with zero indices.
  @inlinable
  public func any() -> Bool {
    let axes = Tensor<Int32>(rangeFrom: 0, to: Int32(rank), stride: 1, on: device)
    return _Raw.any(self, reductionIndices: axes).scalarized()
  }

  /// Performs a logical AND operation along the specified axes. The reduced dimensions are
  /// removed.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  public func all(squeezingAxes axes: Int...) -> Tensor {
    ensureValid(axes: axes)
    let axes = axes.map(Int32.init)
    return _Raw.all(self, reductionIndices: Tensor<Int32>(axes, on: device), keepDims: false)
  }

  /// Performs a logical AND operation along the specified axes. The reduced dimensions are
  /// removed.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  public func any(squeezingAxes axes: Int...) -> Tensor {
    ensureValid(axes: axes)
    let axes = axes.map(Int32.init)
    return _Raw.any(self, reductionIndices: Tensor<Int32>(axes, on: device), keepDims: false)
  }

  /// Performs a logical AND operation along the specified axes. The reduced dimensions are
  /// retained with value 1.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  public func all(alongAxes axes: Int...) -> Tensor {
    ensureValid(axes: axes)
    let axes = axes.map(Int32.init)
    return _Raw.all(self, reductionIndices: Tensor<Int32>(axes, on: device), keepDims: true)
  }

  /// Performs a logical OR operation along the specified axes. The reduced
  /// dimensions are retained with value 1.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  public func any(alongAxes axes: Int...) -> Tensor {
    ensureValid(axes: axes)
    let axes = axes.map(Int32.init)
    return _Raw.any(self, reductionIndices: Tensor<Int32>(axes, on: device), keepDims: true)
  }
}
//
//extension Tensor where Scalar: Numeric & Comparable { dimensions
//  /// are removed.
//  /// - Parameter axes: The dimensions to reduce.
//  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//  @inlinable
//  public func argmax(squeezingAxis axis: Int) -> Tensor<Int32> {
//    ensureValid(axes: [axis])
//    return _Raw.argMax(self, dimension: Int64(axis))
//  }
//
//  /// Returns the indices of the minimum values along the specified axes. The reduced dimensions
//  /// are removed.
//  /// - Parameter axes: The dimensions to reduce.
//  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//  @inlinable
//  public func argmin(squeezingAxis axis: Int) -> Tensor<Int32> {
//    ensureValid(axes: [axis])
//    return _Raw.argMin(self, dimension: Tensor<Int32>(Int32(axis), on: device))
//  }
//  /// Returns the index of the maximum value of the flattened scalars.
//  @inlinable
//  public func argmax() -> Tensor<Int32> {
//    flattened().argmax(squeezingAxis: 0)
//  }
//
//  /// Returns the index of the minimum value of the flattened scalars.
//  @inlinable
//  public func argmin() -> Tensor<Int32> {
//    flattened().argmin(squeezingAxis: 0)
//  }
//}

//extension Tensor where Scalar: TensorFlowFloatingPoint {
//  // Note: adapted from `_MinOrMaxGrad`:
//  // https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/math_grad.py#L223.
//  @inlinable
//  func _vjpMinMaxHelper(
//    squeezingAxes axes: Tensor<Int32>,
//    originalValue: Tensor,
//    seed: Tensor
//  ) -> Tensor {
//    fatalError()
//  }
//
//  @inlinable
//  @derivative(of: max(squeezingAxes:))
//  func _vjpMax(squeezingAxes axes: Tensor<Int32>) -> (
//    value: Tensor, pullback: (Tensor) -> Tensor
//  ) {
//    let result = max(squeezingAxes: axes)
//    return (
//      result,
//      { v in
//        self._vjpMinMaxHelper(squeezingAxes: axes, originalValue: result, seed: v)
//      }
//    )
//  }
//
//  @inlinable
//  @derivative(of: min(squeezingAxes:))
//  func _vjpMin(squeezingAxes axes: Tensor<Int32>) -> (
//    value: Tensor, pullback: (Tensor) -> Tensor
//  ) {
//    let result = min(squeezingAxes: axes)
//    return (
//      result,
//      { v in
//        self._vjpMinMaxHelper(squeezingAxes: axes, originalValue: result, seed: v)
//      }
//    )
//  }
//
//  // Note: adapted from `_MinOrMaxGrad`:
//  // https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/math_grad.py#L223.
//  @inlinable
//  func _vjpMinMaxHelper(
//    alongAxes axes: Tensor<Int32>,
//    originalValue: Tensor,
//    seed: Tensor
//  ) -> Tensor {
//    fatalError()
//  }
//
//  @inlinable
//  @derivative(of: max(alongAxes:))
//  func _vjpMax(alongAxes axes: Tensor<Int32>) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
//    let result = max(alongAxes: axes)
//    return (
//      result,
//      { v in
//        self._vjpMinMaxHelper(alongAxes: axes, originalValue: result, seed: v)
//      }
//    )
//  }
//
//  @inlinable
//  @derivative(of: min(alongAxes:))
//  func _vjpMin(alongAxes axes: Tensor<Int32>) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
//    let result = min(alongAxes: axes)
//    return (
//      result,
//      { v in
//        self._vjpMinMaxHelper(alongAxes: axes, originalValue: result, seed: v)
//      }
//    )
//  }
//}

// MARK: - Numeric Reductions

extension Tensor where Scalar: Numeric {
  // MARK: - Sum

  /// Returns the sum along the specified axes. The reduced dimensions are removed.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func sum(squeezingAxes axes: Tensor<Int32>) -> Tensor {
    ensureValid(axes: axes)
    return _Raw.sum(self, reductionIndices: axes.scalars.map { Int64($0) }, keepDims: false)
  }

  /// Returns the sum along the specified axes. The reduced dimensions are removed.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func sum(squeezingAxes axes: [Int]) -> Tensor {
    let axes = axes.map(Int64.init)
    return _Raw.sum(self, reductionIndices: axes, keepDims: false)
  }

  /// Returns the sum along the specified axes. The reduced dimensions are removed.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank...rank`.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func sum(squeezingAxes axes: Int...) -> Tensor {
    sum(squeezingAxes: axes)
  }

  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func sum() -> Tensor {
    flattened().sum(squeezingAxes: 0)
  }

  /// Returns the sum along the specified axes. The reduced dimensions are retained with value 1.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func sum(alongAxes axes: Tensor<Int32>) -> Tensor {
    ensureValid(axes: axes)
    return _Raw.sum(self, reductionIndices: axes, keepDims: true)
  }

  /// Returns the sum along the specified axes. The reduced dimensions are retained with value 1.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func sum(alongAxes axes: [Int]) -> Tensor {
    let axes = axes.map(Int64.init)
    return _Raw.sum(self, reductionIndices: axes, keepDims: true)
  }

  /// Returns the sum along the specified axes. The reduced dimensions are retained with value 1.
  /// - Parameter axes: The dimensions to reduce.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func sum(alongAxes axes: Int...) -> Tensor {
    sum(alongAxes: axes)
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: sum(alongAxes:))
  func _vjpSum(alongAxes axes: Tensor<Int32>) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    return _vjpSum(alongAxes: axes.scalars.map { Int($0) })
  }

  @inlinable
  @derivative(of: sum(squeezingAxes:))
  func _vjpSum(squeezingAxes axes: Tensor<Int32>) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    return _vjpSum(squeezingAxes: axes.scalars.map { Int($0) })
  }

  @inlinable
  @derivative(of: sum(alongAxes:))
  func _vjpSum(alongAxes axes: [Int]) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let value = sum(alongAxes: axes)
    let m = shape.dimensions.map { Int64($0) }
    return (value, { _Raw.broadcastTo($0, shape: m) })
  }

  @inlinable
  @derivative(of: sum(squeezingAxes:))
  func _vjpSum(squeezingAxes axes: [Int]) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let value = sum(squeezingAxes: axes)
    let rank = self.rank
    return (
      value,
      { [shape = shape.dimensions.map { Int64($0) }] v in
        var expandedShape = shape
        for dim in axes { expandedShape[(dim + rank) % rank] = 1 }
        let unsqueezed = _Raw.reshape(v, shape: expandedShape)
        return _Raw.broadcastTo(unsqueezed, shape: shape)
      }
    )
  }
}
