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

infix operator .==: ComparisonPrecedence
infix operator .!=: ComparisonPrecedence

/// Special protocol for calling tensorflow operations that take heterogeneous arrays as input.
public protocol AnyTensor {
  var _rawTensorHandle: CTensorHandle { get }
  var _tensorFlowDataType: TensorDataType { get }
  var scalarType: TensorFlowScalar.Type { get }
}


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
  public var _rawTensorHandle: CTensorHandle { return handle._cTensorHandle }
  public var _tensorFlowDataType: TensorDataType { return Scalar.tensorFlowDataType }
  public var scalarType: TensorFlowScalar.Type { return Scalar.self }
}

extension Tensor {
  /// The number of dimensions of the `Tensor`.
  public var rank: Int {
    @_semantics("autodiff.nonvarying")
    get { handle.rank }
  }

  /// The shape of the `Tensor`.
  public var shape: TensorShape {
    @_semantics("autodiff.nonvarying")
    get { handle.shape }
  }

  /// The number of scalars in the `Tensor`.
  @inlinable
  public var scalarCount: Int {
    @_semantics("autodiff.nonvarying")
    get { shape.contiguousSize }
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
  /// Creates a 0-D tensor from a scalar value.
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public init(_ value: Scalar/*, on device: Device = .default*/) {
    fatalError()
//    switch device.backend {
//    case .XLA:
//      self.init(_xla: XLATensor.make(value/*, on: device*/))
//    case .TF_EAGER:
//      self.init(shape: [], scalars: [value]/*, on: device*/)
//    }
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: init(_:))
  static func _vjpScalarInit(_ value: __owned Scalar/*, on device: Device = .default*/) -> (
    value: Tensor, pullback: (Tensor) -> Scalar
  ) {
    fatalError()

  }
}

extension Tensor {
  /// Creates a 1D tensor from scalars.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public init(_ scalars: [Scalar]/*, on device: Device = .default*/) {
    self.init(shape: [scalars.count], scalars: scalars/*, on: device*/)
  }

  /// Creates a 1D tensor from scalars.
  @inlinable
  public init<C: Collection>(
    _ vector: C//, on device: Device = .default
  ) where C.Element == Scalar {
    fatalError()

  }

  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public init(shape: TensorShape, scalars: [Scalar]/*, on device: Device = .default*/) {
    fatalError()

  }


  public init(
    shape: TensorShape,
    scalars: UnsafeBufferPointer<Scalar>//,
    //on device: Device = .default
  ) {
    fatalError()

  }


  @inlinable
  public init(
    shape: TensorShape,
    scalars: [Scalar],
    toReducedPrecision: Bool//,
    //directlyOn device: Device
  ) {
    fatalError()

  }


  public init(
    shape: TensorShape,
    scalars: UnsafeBufferPointer<Scalar>,
    toReducedPrecision: Bool//,
    //directlyOn device: Device
  ) {
    fatalError()

  }

  public init<C: Collection>(
    shape: TensorShape, scalars: C//, on device: Device = .default
  ) where C.Element == Scalar {
    fatalError()

  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: init(_:))
  static func _vjpInit(_ scalars: [Scalar]/*, on device: Device = .default*/) -> (
    value: Tensor, pullback: (Tensor) -> Array<Scalar>.TangentVector
  ) {
    fatalError()

  }

  @inlinable
  @derivative(of: init(shape:scalars:))
  static func _vjpInit(
    shape: TensorShape, scalars: [Scalar]//, on device: Device = .default
  ) -> (value: Tensor, pullback: (Tensor) -> Array<Scalar>.TangentVector) {
    fatalError()

  }
}


@frozen
public struct _TensorElementLiteral<Scalar> where Scalar: TensorFlowScalar {
  @usableFromInline let tensor: Tensor<Scalar>
}

extension _TensorElementLiteral: ExpressibleByBooleanLiteral
where Scalar: ExpressibleByBooleanLiteral {
  public typealias BooleanLiteralType = Scalar.BooleanLiteralType
  @inlinable
  public init(booleanLiteral: BooleanLiteralType) {
    tensor = Tensor(Scalar(booleanLiteral: booleanLiteral))
  }
}

extension _TensorElementLiteral: ExpressibleByIntegerLiteral
where Scalar: ExpressibleByIntegerLiteral {
  public typealias IntegerLiteralType = Scalar.IntegerLiteralType
  @inlinable
  public init(integerLiteral: IntegerLiteralType) {
    tensor = Tensor(Scalar(integerLiteral: integerLiteral))
  }
}

extension _TensorElementLiteral: ExpressibleByFloatLiteral
where Scalar: ExpressibleByFloatLiteral {
  public typealias FloatLiteralType = Scalar.FloatLiteralType
  @inlinable
  public init(floatLiteral: FloatLiteralType) {
    tensor = Tensor(Scalar(floatLiteral: floatLiteral))
  }
}

extension _TensorElementLiteral: ExpressibleByArrayLiteral {
  public typealias ArrayLiteralElement = _TensorElementLiteral<Scalar>
  @inlinable
  public init(arrayLiteral elements: _TensorElementLiteral<Scalar>...) {
    fatalError()

  }
}



//===------------------------------------------------------------------------------------------===//
// Equatable
//===------------------------------------------------------------------------------------------===//

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
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public static func + (lhs: Tensor, rhs: Tensor) -> Tensor {
    fatalError()

  }

  /// Subtracts one tensor from another and produces their difference.
  /// - Note: `-` supports broadcasting.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public static func - (lhs: Tensor, rhs: Tensor) -> Tensor {
    fatalError()

  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: +)
  static func _vjpAdd(lhs: Tensor, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    fatalError()

  }

  @inlinable
  @derivative(of: -)
  static func _vjpSubtract(lhs: Tensor, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    fatalError()

  }
}

extension Tensor: Differentiable /*& EuclideanDifferentiable*/ where Scalar: TensorFlowFloatingPoint {
  public typealias TangentVector = Tensor

  public var zeroTangentVectorInitializer: () -> TangentVector {
    let shape = self.shape
    return { Tensor(zeros: shape) }
  }
}

//extension Tensor {
//  /// The device on which `self` is allocated.
//  public var device: Device {
//    @_semantics("autodiff.nonvarying")
//    get {
//      switch handle.backend {
//      case .XLA:
//        return xlaTensor.device
//      case .TF_EAGER:
//        return Device.defaultTFEager
//      }
//    }
//  }
//}

public protocol TensorProtocol {
  associatedtype Scalar: TensorFlowScalar

  var shape: TensorShape { get }
//  var summary: String { get }
}

public protocol DifferentiableTensorProtocol:
  TensorProtocol & Differentiable //& EuclideanDifferentiable
where Scalar: TensorFlowFloatingPoint {

}

