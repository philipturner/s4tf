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
import CTensorFlow
import Darwin

infix operator .==: ComparisonPrecedence
infix operator .!=: ComparisonPrecedence

/// Special protocol for calling tensorflow operations that take heterogeneous arrays as input.
public protocol AnyTensor {
  var _rawTensorHandle: CTensorHandle { get }
  var _tensorFlowDataType: TensorDataType { get }
  var scalarType: TensorFlowScalar.Type { get }
}

/// A multidimensional array of elements that is a generalization of vectors and matrices to
/// potentially higher dimensions.
///
/// The generic parameter `Scalar` describes the type of scalars in the tensor (such as `Int32`,
///  `Float`, etc).
@frozen
public struct Tensor<Scalar: TensorFlowScalar> {
  /// The underlying `TensorHandle`.
  /// - Note: `handle` is public to allow user defined ops, but should not normally be used.
  public let handle: TensorHandle<Scalar>

  /// An internal marker to identify scalar zero tensors, for use in optimizations.
  @usableFromInline
  internal var _isScalarZero = false

  @inlinable
  public init(handle: TensorHandle<Scalar>) {
    self.handle = handle
  }
}

extension Tensor: AnyTensor {
  public var _rawTensorHandle: CTensorHandle { fatalError() }//return handle._cTensorHandle }
  public var _tensorFlowDataType: TensorDataType { return Scalar.tensorFlowDataType }
  public var scalarType: TensorFlowScalar.Type { return Scalar.self }
}

//===------------------------------------------------------------------------------------------===//
// Tensor Properties
//===------------------------------------------------------------------------------------------===//

extension Tensor {

  /// The number of scalars in the `Tensor`.
  @inlinable
  public var scalarCount: Int {
    @_semantics("autodiff.nonvarying")
    get {  fatalError() }
  }

  /// The rank of the tensor, represented as a `Tensor<Int32>`.
  @inlinable
  public var rankTensor: Tensor<Int32> {
    @_semantics("autodiff.nonvarying")
    get {
      fatalError()
    }
  }

  /// The dimensions of the tensor, represented as a `Tensor<Int32>`.
  @inlinable
  public var shapeTensor: Tensor<Int32> {
    @_semantics("autodiff.nonvarying")
    get {
      fatalError()
    }
  }

  /// The number of scalars in the tensor, represented as a `Tensor<Int32>`.
  @inlinable
  public var scalarCountTensor: Tensor<Int32> {
    @_semantics("autodiff.nonvarying")
    get {
      fatalError()
    }
  }
}

extension Tensor {
  @inlinable
  public var array: ShapedArray<Scalar> {
    fatalError()

  }

  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public var scalars: [Scalar] {
    fatalError()
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: scalars)
  func _vjpScalars() -> (value: [Scalar], pullback: (Array<Scalar>.TangentVector) -> Tensor) {
    fatalError()
//    )
  }
}

extension Tensor {
  /// Creates a 0-D tensor from a scalar value.
//  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public init(_ value: Scalar/*, on device: Device = .default*/) {
    fatalError()
  }
}


extension Tensor {
  /// Creates a 1D tensor from scalars.
  @inlinable
  public init(_ scalars: [Scalar]/*, on device: Device = .default*/) {
    self.init(shape: [scalars.count], scalars: scalars/*, on: device*/)
  }


  @inlinable
  public init(shape: TensorShape, scalars: [Scalar]) {
    precondition(
      shape.contiguousSize == scalars.count,
      """
      The shape requires \(shape.contiguousSize) scalars but \(scalars.count) were \
      provided.
      """)
    self = scalars.withUnsafeBufferPointer { bufferPointer in
      Tensor(shape: shape, scalars: bufferPointer/*, on: device*/)
    }
  }

  public init(
    shape: TensorShape,
    scalars: UnsafeBufferPointer<Scalar>
  ) {
    fatalError()
  }

  @inlinable
  public init(
    shape: TensorShape,
    scalars: [Scalar],
    toReducedPrecision: Bool
  ) {
    precondition(
      shape.contiguousSize == scalars.count,
      """
      The shape requires \(shape.contiguousSize) scalars but \(scalars.count) were \
      provided.
      """)
    self = scalars.withUnsafeBufferPointer { bufferPointer in
      Tensor(
        shape: shape, scalars: bufferPointer, toReducedPrecision: toReducedPrecision)
    }
  }

  public init(
    shape: TensorShape,
    scalars: UnsafeBufferPointer<Scalar>,
    toReducedPrecision: Bool
  ) {
    fatalError()
  }

  public init<C: Collection>(
    shape: TensorShape, scalars: C
  ) where C.Element == Scalar {
    precondition(
      shape.contiguousSize == scalars.count,
      """
      The shape requires \(shape.contiguousSize) scalars but \(scalars.count) were \
      provided.
      """)
    self.init(shape: shape, scalars: [Scalar](scalars)/*, on: device*/)
  }
}

extension Tensor: Equatable where Scalar: Equatable {
  @inlinable
  public static func == (lhs: Tensor, rhs: Tensor) -> Bool {
    fatalError()
  }

  @inlinable
  public static func != (lhs: Tensor, rhs: Tensor) -> Bool {
    fatalError()
  }
}


extension Tensor: AdditiveArithmetic where Scalar: Numeric {
  /// The scalar zero tensor.
  public static var zero: Tensor {
    fatalError()
  }

  @inlinable
  public static func + (lhs: Tensor, rhs: Tensor) -> Tensor {
    fatalError()
  }

  @inlinable
  public static func - (lhs: Tensor, rhs: Tensor) -> Tensor {
    fatalError()
  }
}



extension Tensor: Differentiable where Scalar: TensorFlowFloatingPoint {
  public typealias TangentVector = Tensor
}
