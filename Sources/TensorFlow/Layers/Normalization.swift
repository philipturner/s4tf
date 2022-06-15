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

@_semantics("autodiff.nonvarying")
func withoutDerivative<T, R>(at x: T, in body: (T) -> R) -> R {
  fatalError()
}

struct BatchNorm<Scalar> {
  @differentiable(reverse)
  func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    doInference(input)
  }

  func doInference(
    _ input: Tensor<Scalar>
  ) -> Tensor<Scalar> {
    let eps = withoutDerivative(at: input) { $0 }
    return eps
  }
}

struct LayerNorm<Scalar> {
  @differentiable(reverse)
  func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
//    input + Tensor<Scalar>.sqrt(input)
    Tensor<Scalar>.add2(input, Tensor<Scalar>.sqrt(input))
  }
}
