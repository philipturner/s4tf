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
  public init(_ value: Scalar, on device: Device = .default) {
    switch device.backend {
    case .XLA:
      self.init(_xla: XLATensor.make(value, on: device))
    case .TF_EAGER:
      self.init(shape: [], scalars: [value], on: device)
    }
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: init(_:on:))
  static func _vjpScalarInit(_ value: __owned Scalar, on device: Device = .default) -> (
    value: Tensor, pullback: (Tensor) -> Scalar
  ) {
    fatalError()
//    return (Tensor(value, on: device), { $0.scalarized() })
  }
}

extension Tensor {
  /// Creates a 1D tensor from scalars.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public init(_ scalars: [Scalar], on device: Device = .default) {
    self.init(shape: [scalars.count], scalars: scalars, on: device)
  }

  /// Creates a 1D tensor from scalars.
  @inlinable
  public init<C: Collection>(
    _ vector: C, on device: Device = .default
  ) where C.Element == Scalar {
    fatalError()
//    self.init([Scalar](vector), on: device)
  }

  /// Creates a tensor with the specified shape and contiguous scalars in row-major order.
  ///
  /// - Parameters:
  ///   - shape: The shape of the tensor.
  ///   - scalars: The scalar contents of the tensor.
  /// - Precondition: The product of the dimensions of the shape must equal the number of scalars.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public init(shape: TensorShape, scalars: [Scalar], on device: Device = .default) {
    precondition(
      shape.contiguousSize == scalars.count,
      """
      The shape requires \(shape.contiguousSize) scalars but \(scalars.count) were \
      provided.
      """)
    self = scalars.withUnsafeBufferPointer { bufferPointer in
      Tensor(shape: shape, scalars: bufferPointer, on: device)
    }
  }

  /// Creates a tensor with the specified shape and contiguous scalars in row-major order.
  ///
  /// - Parameters:
  ///   - shape: The shape of the tensor.
  ///   - scalars: The scalar contents of the tensor.
  /// - Precondition: The product of the dimensions of the shape must equal the number of scalars.
  public init(
    shape: TensorShape,
    scalars: UnsafeBufferPointer<Scalar>,
    on device: Device = .default
  ) {
    precondition(
      shape.contiguousSize == scalars.count,
      """
      The shape requires \(shape.contiguousSize) scalars but \(scalars.count) were \
      provided.
      """)
    switch device.backend {
    case .XLA:
      self.init(_xla: XLATensor.make(scalars, shape.dimensions, on: device))
    case .TF_EAGER:
      let handle = TensorHandle<Scalar>(
        shape: shape.dimensions,
        scalarsInitializer: { address in
          address.initialize(from: scalars.baseAddress!, count: shape.contiguousSize)
        })
      self.init(handle: handle)
    }
  }

  /// Creates a tensor with the specified shape and contiguous scalars in row-major order.
  ///
  /// - Parameters:
  ///   - shape: The shape of the tensor.
  ///   - scalars: The scalar contents of the tensor.
  /// - Precondition: The product of the dimensions of the shape must equal the number of scalars.
  @inlinable
  public init(
    shape: TensorShape,
    scalars: [Scalar],
    toReducedPrecision: Bool,
    directlyOn device: Device
  ) {
    precondition(
      shape.contiguousSize == scalars.count,
      """
      The shape requires \(shape.contiguousSize) scalars but \(scalars.count) were \
      provided.
      """)
    self = scalars.withUnsafeBufferPointer { bufferPointer in
      Tensor(
        shape: shape, scalars: bufferPointer, toReducedPrecision: toReducedPrecision,
        directlyOn: device)
    }
  }

  /// Creates a tensor with the specified shape and contiguous scalars in row-major order.
  ///
  /// - Parameters:
  ///   - shape: The shape of the tensor.
  ///   - scalars: The scalar contents of the tensor.
  /// - Precondition: The product of the dimensions of the shape must equal the number of scalars.
  public init(
    shape: TensorShape,
    scalars: UnsafeBufferPointer<Scalar>,
    toReducedPrecision: Bool,
    directlyOn device: Device
  ) {
    precondition(
      shape.contiguousSize == scalars.count,
      """
      The shape requires \(shape.contiguousSize) scalars but \(scalars.count) were \
      provided.
      """)
    switch device.backend {
    case .XLA:
      self.init(
        _xla: XLATensor.make(
          scalars, shape.dimensions, toReducedPrecision: toReducedPrecision,
          directlyOn: device))
    case .TF_EAGER:
      precondition(!toReducedPrecision)
      self = .init(shape: shape, scalars: scalars, on: device)
    }
  }

  /// Creates a tensor with the specified shape and contiguous scalars in row-major order.
  ///
  /// - Parameters:
  ///   - shape: The shape of the tensor.
  ///   - scalars: The scalar contents of the tensor.
  /// - Precondition: The product of the dimensions of the shape must equal the number of scalars.
  public init<C: Collection>(
    shape: TensorShape, scalars: C, on device: Device = .default
  ) where C.Element == Scalar {
    precondition(
      shape.contiguousSize == scalars.count,
      """
      The shape requires \(shape.contiguousSize) scalars but \(scalars.count) were \
      provided.
      """)
    self.init(shape: shape, scalars: [Scalar](scalars), on: device)
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: init(_:on:))
  static func _vjpInit(_ scalars: [Scalar], on device: Device = .default) -> (
    value: Tensor, pullback: (Tensor) -> Array<Scalar>.TangentVector
  ) {
    fatalError()

  }

  @inlinable
  @derivative(of: init(shape:scalars:on:))
  static func _vjpInit(
    shape: TensorShape, scalars: [Scalar], on device: Device = .default
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
//    guard lhs.shape == rhs.shape else {
//      return false
//    }
//    return (lhs .== rhs).all()
  }

  @inlinable
  public static func != (lhs: Tensor, rhs: Tensor) -> Bool {
    fatalError()
//    guard lhs.shape == rhs.shape else {
//      return true
//    }
//    return (lhs .!= rhs).any()
  }
}

//===------------------------------------------------------------------------------------------===//
// Description and Visualization
//===------------------------------------------------------------------------------------------===//

//// String conversion.
//extension Tensor: CustomStringConvertible {
//  /// A textual representation of the tensor.
//  ///
//  /// - Note: use `fullDescription` for a non-pretty-printed description showing all scalars.
//  public var description: String {
//    @_semantics("autodiff.nonvarying")
//    get {
//      return array.description
//    }
//  }
//}

//extension Tensor {
//  /// A textual representation of the tensor. Returns a summarized description if `summarize` is
//  /// true and the element count exceeds twice the `edgeElementCount`.
//  ///
//  /// - Parameters:
//  ///   - lineWidth: The max line width for printing. Used to determine number of scalars to print
//  ///     per line.
//  ///   - edgeElementCount: The maximum number of elements to print before and after summarization
//  ///     via ellipses (`...`).
//  ///   - summarizing: If true, summarize description if element count exceeds twice
//  ///     `edgeElementCount`.
//  public func description(
//    lineWidth: Int = 80,
//    edgeElementCount: Int = 3,
//    summarizing: Bool = false
//  ) -> String {
//    return array.description(
//      lineWidth: lineWidth,
//      edgeElementCount: edgeElementCount,
//      summarizing: summarizing)
//  }
//
//  /// A full, non-pretty-printed textual representation of the tensor, showing
//  /// all scalars.
//  public var fullDescription: String {
//    @_semantics("autodiff.nonvarying")
//    get {
//      return array.fullDescription
//    }
//  }
//
//  public var irText: String { XLATensor.irText(xlaTensor) }
//}

// Xcode Playground display conversion.
//extension Tensor: CustomPlaygroundDisplayConvertible {
//  public var playgroundDescription: Any {
//    @_semantics("autodiff.nonvarying")
//    get {
//      return description
//    }
//  }
//}
//
//// Mirror representation, used by debugger/REPL.
//extension Tensor: CustomReflectable {
//  public var customMirror: Mirror {
//    @_semantics("autodiff.nonvarying")
//    get {
//      return Mirror(self, children: [], displayStyle: .struct)
//    }
//  }
//}

//===------------------------------------------------------------------------------------------===//
// Codable Conformance
//===------------------------------------------------------------------------------------------===//

//extension Tensor: Codable where Scalar: Codable {
//  @inlinable
//  public func encode(to encoder: Encoder) throws {
//    var container = encoder.singleValueContainer()
//    try container.encode(array)
//  }
//
//  @inlinable
//  public init(from decoder: Decoder) throws {
//    let container = try decoder.singleValueContainer()
//    let array = try container.decode(ShapedArray<Scalar>.self)
//    self.init(array)
//  }
//}

//===------------------------------------------------------------------------------------------===//
// Additive Group
//===------------------------------------------------------------------------------------------===//

extension Tensor: AdditiveArithmetic where Scalar: Numeric {
  /// The scalar zero tensor.
  public static var zero: Tensor {
    fatalError()
//    var zero = Tensor(0, on: _DeviceThreadLocalState.local.currentDevice)
//    if _DeviceThreadLocalState.local.isReducedPrecision {
//      zero = zero.toReducedPrecision
//    }
//    zero._isScalarZero = true
//    return zero
  }

  /// Adds two tensors and produces their sum.
  /// - Note: `+` supports broadcasting.
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
//    (
//      lhs + rhs,
//      { [broadcastPb = BroadcastingPullback(lhs, rhs)] v in
//        return broadcastPb(v, v)
//      }
//    )
  }

  @inlinable
  @derivative(of: -)
  static func _vjpSubtract(lhs: Tensor, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    fatalError()
//    (
//      lhs - rhs,
//      { [broadcastPb = BroadcastingPullback(lhs, rhs)] v in
//        return broadcastPb(v, -v)
//      }
//    )
  }
}


//===------------------------------------------------------------------------------------------===//
// Differentiable
//===------------------------------------------------------------------------------------------===//

extension Tensor: Differentiable /*& EuclideanDifferentiable*/ where Scalar: TensorFlowFloatingPoint {
  public typealias TangentVector = Tensor

  public var zeroTangentVectorInitializer: () -> TangentVector {
    let shape = self.shape
    return { Tensor(zeros: shape) }
  }
}

//===------------------------------------------------------------------------------------------===//
// Multi-device support
//===------------------------------------------------------------------------------------------===//

extension Tensor {
  /// The device on which `self` is allocated.
  public var device: Device {
    @_semantics("autodiff.nonvarying")
    get {
      switch handle.backend {
      case .XLA:
        return xlaTensor.device
      case .TF_EAGER:
        return Device.defaultTFEager
      }
    }
  }
}

//===------------------------------------------------------------------------------------------===//
// Annotations
//===------------------------------------------------------------------------------------------===//

public protocol TensorProtocol {
  associatedtype Scalar: TensorFlowScalar
//  init(repeating repeatedValue: Scalar, shape: TensorShape, on device: Device)
//  var annotations: String { get }
  var shape: TensorShape { get }
//  var summary: String { get }
}

public protocol DifferentiableTensorProtocol:
  TensorProtocol & Differentiable //& EuclideanDifferentiable
where Scalar: TensorFlowFloatingPoint {
//  @differentiable(reverse, wrt: self)
//  func annotate(_ annotation: String) -> Self
}

//extension Tensor: TensorProtocol {
//  /// The annotations describing this tensor.
//  public var annotations: String {
//    switch handle.backend {
//    case .XLA:
//      return XLATensor.annotations(xlaTensor)
//    case .TF_EAGER:
//      return Device.defaultTFEager.annotationsAvailable
//    }
//  }
//
//  /// An alias for annotations.
//  public var summary: String { annotations }
//}

//extension Tensor: DifferentiableTensorProtocol
//where Scalar: TensorFlowFloatingPoint {
//  /// Adds an annotation.
//  ///
//  /// Note: Only X10 is supported. For other backends, umodified `self` is
//  /// returned.
//  ///
//  /// - Parameter annotation: The annotation to be added.
//  /// - Returns: The annotated tensor.
//  @differentiable(reverse, wrt: self)
//  public func annotate(_ annotation: String) -> Tensor<Scalar> {
//    switch handle.backend {
//    case .XLA:
//      return Tensor<Scalar>(_xla: XLATensor.annotate(xlaTensor, annotation))
//    case .TF_EAGER:
//      return self
//    }
//  }
//
//  @derivative(of: annotate)
//  @usableFromInline
//  func vjpAnnotate(_ annotation: String) -> (
//    value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> Tensor<Scalar>
//  ) {
//    (annotate(annotation), { $0 })
//  }
//}
