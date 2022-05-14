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

extension Tensor: VectorProtocol where Scalar: TensorFlowFloatingPoint {
  public typealias VectorSpaceScalar = Float

  public func scaled(by scale: Float) -> Self {
    Scalar(scale) * self
  }

  public func adding(_ scalar: Float) -> Self {
    self + Scalar(scalar)
  }

  public func subtracting(_ scalar: Float) -> Self {
    self - Scalar(scalar)
  }
}

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
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: +)
  static func _vjpAdd(lhs: Tensor, rhs: Scalar) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Scalar)
  ) {
    fatalError()
  }

  @inlinable
  @derivative(of: +)
  static func _vjpAdd(lhs: Scalar, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Scalar, Tensor)
  ) {
    fatalError()
  }

  @inlinable
  @derivative(of: -)
  static func _vjpSubtract(lhs: Tensor, rhs: Scalar) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Scalar)
  ) {
    fatalError()
  }

  @inlinable
  @derivative(of: -)
  static func _vjpSubtract(lhs: Scalar, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Scalar, Tensor)
  ) {
    fatalError()
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
  }

  @inlinable
  @derivative(of: *)
  static func _vjpMultiply(lhs: Scalar, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Scalar, Tensor)
  ) {
    fatalError()
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
}

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
