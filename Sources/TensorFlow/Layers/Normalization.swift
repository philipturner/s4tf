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

@differentiable(reverse, wrt: (input, mean, variance, offset, scale))
private func normalize<Scalar: TensorFlowFloatingPoint>(
  _ input: Tensor<Scalar>,
  mean: Tensor<Scalar>,
  variance: Tensor<Scalar>,
  offset: Tensor<Scalar>,
  scale: Tensor<Scalar>,
  varianceEpsilon: Tensor<Scalar>
) -> Tensor<Scalar> {
  return input + mean + variance + offset + scale
}

public struct BatchNorm<Scalar: TensorFlowFloatingPoint>: Layer {
  @noDerivative public let axis: Int
  @noDerivative public let momentum: Scalar
  public var offset: Tensor<Scalar>
  public var scale: Tensor<Scalar>
  @noDerivative public let epsilon: Scalar
  @noDerivative public var runningMean: Parameter<Scalar>
  @noDerivative public var runningVariance: Parameter<Scalar>

  @differentiable(reverse)
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let inputRank = input.rank
    let positiveAxis = (inputRank + axis) % inputRank
    let offsetOriginal = self.offset
    let scaleOriginal = self.scale
    let (offset, scale) = Self._sr13263workaround(offset: offsetOriginal,
                                                  scale: scaleOriginal,
                                                  input: input,
                                                  positiveAxis: positiveAxis)
    switch Context.local.learningPhase {
    case .training:
      return doTraining(input, offset: offset, scale: scale, axis: positiveAxis)
    case .inference:
      return doInference(input, offset: offset, scale: scale)
    }
  }
  
  @inline(never)
  @differentiable(reverse) // if the function is `public` or `internal`, the compiler crashes
  private static func _sr13263workaround(
    offset: Tensor<Scalar>,
    scale: Tensor<Scalar>,
    input: Tensor<Scalar>,
    positiveAxis: Int
  ) -> (Tensor<Scalar>, Tensor<Scalar>) {
    if positiveAxis != input.rank - 1 {
      var broadcastShape = TensorShape([Int](repeating: 1, count: input.rank))
      broadcastShape[positiveAxis] = input.shape[positiveAxis]
      return (offset.reshaped(to: broadcastShape), scale.reshaped(to: broadcastShape))
    } else {
      return (offset, scale)
    }
  }
  
  private func doTraining(
    _ input: Tensor<Scalar>, offset: Tensor<Scalar>, scale: Tensor<Scalar>, axis: Int
  ) -> Tensor<Scalar> {
//    return input
    var normalizedAxes = Array(0..<input.rank)
    normalizedAxes.remove(at: axis)
    let moments = input.moments(alongAxes: normalizedAxes)
    let decayMomentum = Tensor(1 - momentum)//, on: input.device)
    let isReducedPrecision = withoutDerivative(at: input) { $0.isReducedPrecision }
    var momentsMean = moments.mean
    var momentsVariance = moments.variance
    if isReducedPrecision {
      momentsMean = momentsMean.toFullPrecision
      momentsVariance = momentsVariance.toFullPrecision
    }
    runningMean.value += (momentsMean - runningMean.value) * decayMomentum
    runningVariance.value += (momentsVariance - runningVariance.value) * decayMomentum
    let eps = withoutDerivative(at: input) { Tensor(epsilon, deviceAndPrecisionLike: $0) }
    return eps
//    return normalize(
//      input,
//      mean: moments.mean, variance: moments.variance,
//      offset: offset, scale: scale,
//      varianceEpsilon: eps)
  }

  private func doInference(
    _ input: Tensor<Scalar>, offset: Tensor<Scalar>, scale: Tensor<Scalar>
  ) -> Tensor<Scalar> {
    let isReducedPrecision = withoutDerivative(at: input) { $0.isReducedPrecision }
    let runningVarianceValue =
      isReducedPrecision ? runningVariance.value.toReducedPrecision : runningVariance.value
    let runningMeanValue =
      isReducedPrecision ? runningMean.value.toReducedPrecision : runningMean.value
    let eps = withoutDerivative(at: input) { Tensor(epsilon, deviceAndPrecisionLike: $0) }
    return eps
//    return normalize(
//      input,
//      mean: runningMeanValue, variance: runningVarianceValue,
//      offset: offset, scale: scale,
//      varianceEpsilon: eps)
  }
}

public struct LayerNorm<Scalar: TensorFlowFloatingPoint> {
  public var offset: Tensor<Scalar>
  public var scale: Tensor<Scalar>
  @noDerivative public let axis: Int
  @noDerivative public let epsilon: Scalar

  @differentiable(reverse)
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let epsilon = withoutDerivative(at: input) { Tensor(self.epsilon, deviceAndPrecisionLike: $0) }
    let positiveAxis = (input.rank + axis) % input.rank
    
    var broadcastShape = TensorShape(Array(repeating: 1, count: input.rank))
    broadcastShape[positiveAxis] = input.shape[positiveAxis]
    let offset = self.offset.reshaped(to: broadcastShape)
    let scale = self.scale.reshaped(to: broadcastShape)
    let moments = input.moments(alongAxes: positiveAxis)
    let inv = rsqrt(moments.variance)
    return moments.mean * inv
  }
}
