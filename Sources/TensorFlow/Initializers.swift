//// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
////
//// Licensed under the Apache License, Version 2.0 (the "License");
//// you may not use this file except in compliance with the License.
//// You may obtain a copy of the License at
////
////     http://www.apache.org/licenses/LICENSE-2.0
////
//// Unless required by applicable law or agreed to in writing, software
//// distributed under the License is distributed on an "AS IS" BASIS,
//// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//// See the License for the specific language governing permissions and
//// limitations under the License.

import _Differentiation

//#if !COMPILING_TENSORFLOW_STDLIB_MODULE
//  import Tensor
//#endif

//extension Tensor {
//  /// Creates a tensor with the specified shape and a single, repeated scalar value.
//  ///
//  /// - Parameters:
//  ///   - shape: The dimensions of the tensor.
//  ///   - repeatedValue: The scalar value to repeat.
////  @inlinable

extension Tensor where Scalar: Numeric {
  /// Creates a tensor with all scalars set to zero.
  ///
  /// - Parameter shape: Shape of the tensor.
//  @inlinable
  public init(zeros shape: TensorShape, on device: Device = .default) {
    fatalError()
//    self.init(repeating: 0, shape: shape, on: device)
  }

  /// Creates a tensor with all scalars set to one.
  ///
  /// - Parameter shape: Shape of the tensor.
//  @inlinable
  public init(ones shape: TensorShape, on device: Device = .default) {
    fatalError()
//    self.init(repeating: 1, shape: shape, on: device)
  }


//      offValue: Tensor(offValue, on: device),
//      axis: Int64(axis))
//  }
}
