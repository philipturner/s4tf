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

/// Returns a tensor with the same shape and scalars as the specified tensor.
@inlinable
@differentiable(reverse where Scalar: TensorFlowFloatingPoint)
public func identity<Scalar>(_ x: Tensor<Scalar>) -> Tensor<Scalar> {
  x
}

extension TensorFlowScalar {
  /// Convert to a tensor with the specified rank, with all dimensions equal to `1`.
  @inlinable
  public func makeTensor(rank: Int, on device: Device = .default) -> Tensor<Self> {
    fatalError()
//    return Tensor(repeating: self, shape: TensorShape(rank), on: device)
  }
}
