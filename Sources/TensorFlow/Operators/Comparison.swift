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

infix operator .<: ComparisonPrecedence
infix operator .<=: ComparisonPrecedence
infix operator .>=: ComparisonPrecedence
infix operator .>: ComparisonPrecedence
infix operator .==: ComparisonPrecedence
infix operator .!=: ComparisonPrecedence

extension Tensor where Scalar: Numeric & Comparable {
  /// Returns a tensor of Boolean scalars by computing `lhs < rhs` element-wise.
  @inlinable
  public static func .< (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
    return _Raw.less(lhs, rhs)
  }

  /// Returns a tensor of Boolean scalars by computing `lhs <= rhs` element-wise.
  @inlinable
  public static func .<= (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
    return _Raw.lessEqual(lhs, rhs)
  }

  /// Returns a tensor of Boolean scalars by computing `lhs > rhs` element-wise.
  @inlinable
  public static func .> (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
    return _Raw.greater(lhs, rhs)
  }

  /// Returns a tensor of Boolean scalars by computing `lhs >= rhs` element-wise.
  @inlinable
  public static func .>= (lhs: Tensor, rhs: Tensor) -> Tensor<Bool> {
    return _Raw.greaterEqual(lhs, rhs)
  }

  /// Returns a tensor of Boolean scalars by computing `lhs < rhs` element-wise.
  /// - Note: `.<` supports broadcasting.
  @inlinable
  public static func .< (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
    return _Raw.less(Tensor(lhs, deviceAndPrecisionLike: rhs), rhs)
  }

  /// Returns a tensor of Boolean scalars by computing `lhs <= rhs` element-wise.
  /// - Note: `.<=` supports broadcasting.
  @inlinable
  public static func .<= (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
    return _Raw.lessEqual(Tensor(lhs, deviceAndPrecisionLike: rhs), rhs)
  }

  /// Returns a tensor of Boolean scalars by computing `lhs > rhs` element-wise.
  /// - Note: `.>` supports broadcasting.
  @inlinable
  public static func .> (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
    return _Raw.greater(Tensor(lhs, deviceAndPrecisionLike: rhs), rhs)
  }

  /// Returns a tensor of Boolean scalars by computing `lhs >= rhs` element-wise.
  /// - Note: `.>=` supports broadcasting.
  @inlinable
  public static func .>= (lhs: Scalar, rhs: Tensor) -> Tensor<Bool> {
    return _Raw.greaterEqual(Tensor(lhs, deviceAndPrecisionLike: rhs), rhs)
  }

  /// Returns a tensor of Boolean scalars by computing `lhs < rhs` element-wise.
  /// - Note: `.<` supports broadcasting.
  @inlinable
  public static func .< (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
    return _Raw.less(lhs, Tensor(rhs, deviceAndPrecisionLike: lhs))
  }

  /// Returns a tensor of Boolean scalars by computing `lhs <= rhs` element-wise.
  /// - Note: `.<=` supports broadcasting.
  @inlinable
  public static func .<= (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
    return _Raw.lessEqual(lhs, Tensor(rhs, deviceAndPrecisionLike: lhs))
  }

  /// Returns a tensor of Boolean scalars by computing `lhs > rhs` element-wise.
  /// - Note: `.>` supports broadcasting.
  @inlinable
  public static func .> (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
    return _Raw.greater(lhs, Tensor(rhs, deviceAndPrecisionLike: lhs))
  }

  /// Returns a tensor of Boolean scalars by computing `lhs >= rhs` element-wise.
  /// - Note: `.>=` supports broadcasting.
  @inlinable
  public static func .>= (lhs: Tensor, rhs: Scalar) -> Tensor<Bool> {
    return _Raw.greaterEqual(lhs, Tensor(rhs, deviceAndPrecisionLike: lhs))
  }
}
