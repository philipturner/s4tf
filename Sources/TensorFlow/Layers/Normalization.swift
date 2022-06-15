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

//@differentiable(reverse, wrt: (input, mean, variance, offset, scale))
//private func normalize<Scalar: TensorFlowFloatingPoint>(
//  _ input: Tensor<Scalar>,
//  mean: Tensor<Scalar>,
//  variance: Tensor<Scalar>,
//  offset: Tensor<Scalar>,
//  scale: Tensor<Scalar>,
//  varianceEpsilon: Tensor<Scalar>
//) -> Tensor<Scalar> {
//  return input + mean + variance + offset + scale
//}

public struct BatchNorm<Scalar: TensorFlowFloatingPoint>: Layer {
  public var offset: Tensor<Scalar>
  public var scale: Tensor<Scalar>

  @differentiable(reverse)
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    if true {
      let eps = withoutDerivative(at: input) { Tensor(2.333, deviceAndPrecisionLike: $0) }
      return eps
    } else {
      return doInference(input)
    }
  }
  
  private func doInference(
    _ input: Tensor<Scalar>
  ) -> Tensor<Scalar> {
    let eps = withoutDerivative(at: input) { Tensor(2.333, deviceAndPrecisionLike: $0) }
    return eps
  }
}

public struct LayerNorm<Scalar: TensorFlowFloatingPoint> {
  @differentiable(reverse)
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let moments = input.moments(alongAxes: 1)
    let inv = rsqrt(moments.mean)
    return moments.mean * inv
  }
}
