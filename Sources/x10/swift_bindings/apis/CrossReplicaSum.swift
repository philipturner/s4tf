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

///// Implements crossReplicaSum.
//protocol CrossReplicaSummable {
//  /// A cross replica sum is an operation that runs simultaneously on multiple threads on
//  /// multiple devices and replaces the value on each thread with a sum of all the other values.
//  mutating func crossReplicaSum(_ scale: Double)
//}
//
//extension CrossReplicaSummable {
//  /// Helper that applies a cross replica sum operation to a particular keypath.
//  static func _doCrossReplicaSum<Root>(
//    _ root: inout Root, _ partialKeyPath: PartialKeyPath<Root>,
//    _ scale: Double
//  ) {
//    fatalError()
////    guard let keyPath = partialKeyPath as? WritableKeyPath<Root, Self> else {
////      fatalError("Key path \(partialKeyPath) not writeable cannot copy to device")
////    }
////    root[keyPath: keyPath].crossReplicaSum(scale)
//  }
//}

extension Tensor/*: CrossReplicaSummable*/ where Scalar: TensorFlowNumeric {
  /// Runs a cross replica sum for this tensor. The same cross replica sum
  /// must happen on each of the other devices participating in the sum.
  public mutating func crossReplicaSum(_ scale: Double) {
    fatalError()
  }
}
