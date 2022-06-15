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

extension Tensor {

  /// Reshape to the shape of the specified `Tensor`.
  /// - Precondition: The number of scalars matches the new shape.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func reshaped<T>(like other: Tensor<T>) -> Tensor {
    reshaped(toShape: other.shapeTensor)
  }

  /// Reshape to the specified shape.
  /// - Precondition: The number of scalars matches the new shape.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func reshaped(to newShape: TensorShape) -> Tensor {
    fatalError()
  }

  /// Reshape to the specified `Tensor` representing a shape.
  /// - Precondition: The number of scalars matches the new shape.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func reshaped(toShape newShape: Tensor<Int32>) -> Tensor {
    fatalError()
  }


}

extension Tensor where Scalar: TensorFlowFloatingPoint {
//  @inlinable

  @inlinable
  @derivative(of: reshaped)
  func _vjpReshaped(toShape newShape: Tensor<Int32>) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    fatalError()
//    let value = reshaped(toShape: newShape)
//    return (value, { [shape = shapeTensor] v in v.reshaped(toShape: shape) })
  }

  @inlinable
  @derivative(of: reshaped)
  func _vjpReshaped(toShape newShape: TensorShape) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    fatalError()
//    let value = reshaped(to: newShape)
//    return (value, { [shape = shape] v in v.reshaped(to: shape) })
  }

}
