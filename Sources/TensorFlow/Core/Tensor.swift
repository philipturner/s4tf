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

struct Tensor<Scalar> {}

extension Tensor: Equatable {
  static func == (lhs: Tensor, rhs: Tensor) -> Bool { fatalError() }
  static func != (lhs: Tensor, rhs: Tensor) -> Bool { fatalError() }
}

extension Tensor: AdditiveArithmetic {
  static var zero: Tensor { fatalError() }

  @differentiable(reverse)
  static func + (lhs: Tensor, rhs: Tensor) -> Tensor { fatalError() }
  static func - (lhs: Tensor, rhs: Tensor) -> Tensor { fatalError() }
  
  @differentiable(reverse)
  static func add2(_ lhs: Tensor, _ rhs: Tensor) -> Tensor { fatalError() }
}

extension Tensor {
  @derivative(of: +)
  static func _vjpAdd(lhs: Tensor, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    fatalError()
  }
  
  @derivative(of: add2)
  static func _vjpAdd(_ lhs: Tensor, _ rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    fatalError()
  }
}

extension Tensor: Differentiable  {
  typealias TangentVector = Tensor
}

extension Tensor {
  @differentiable(reverse)
  static func sqrt(_ x: Self) -> Self {
    fatalError()
  }

  @inlinable
  @derivative(of: sqrt)
  static func _vjpSqrt(
    _ x: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    fatalError()
  }
}
