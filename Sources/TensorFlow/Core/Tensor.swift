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

  // /// An internal workaround for SR-13263: debug info generation crash.
  // @usableFromInline
  // class SR13263Workaround {}

  // /// An internal workaround for SR-13263: debug info generation crash.
  // internal var _sr13263Workaround: SR13263Workaround?
  
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

//===------------------------------------------------------------------------------------------===//
// Tensor Properties
//===------------------------------------------------------------------------------------------===//

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
      return _Raw.rank(self)
    }
  }

  /// The dimensions of the tensor, represented as a `Tensor<Int32>`.
  @inlinable
  public var shapeTensor: Tensor<Int32> {
    @_semantics("autodiff.nonvarying")
    get {
      return _Raw.shape(self)
    }
  }

  /// The number of scalars in the tensor, represented as a `Tensor<Int32>`.
  @inlinable
  public var scalarCountTensor: Tensor<Int32> {
    @_semantics("autodiff.nonvarying")
    get {
      return _Raw.size(self)
    }
  }
}

//===------------------------------------------------------------------------------------------===//
// Scalar Conversion
//===------------------------------------------------------------------------------------------===//

extension Tensor {
  /// Returns `true` if `rank` is equal to 0 and `false` otherwise.
  @inlinable
  public var isScalar: Bool {
    return rank == 0
  }

  /// Returns the single scalar element if `rank` is equal to 0 and `nil`
  /// otherwise.
  @inlinable
  public var scalar: Scalar? {
    isScalar ? scalars[0] : nil
  }

  /// Reshape to scalar.
  /// - Precondition: The tensor has exactly one scalar.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public func scalarized() -> Scalar {
    precondition(
      shape.contiguousSize == 1,
      "This tensor must have exactly one scalar but contains \(shape.contiguousSize).")
    return scalars[0]
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: scalarized)
  func _vjpScalarized() -> (value: Scalar, pullback: (Scalar) -> Tensor) {
    let device = self.device
    return (scalarized(), { v in Tensor(v, on: device) })
  }
}

extension TensorFlowScalar {
  @inlinable
  public init?(_ tensor: Tensor<Self>) {
    guard let scalar = tensor.scalar else {
      return nil
    }
    self = scalar
  }
}

//===------------------------------------------------------------------------------------------===//
// Array Conversion
//===------------------------------------------------------------------------------------------===//

extension Tensor {
  @inlinable
  public var array: ShapedArray<Scalar> {
    debugLog("Returning a host copy of array.")
    if handle.backend == .XLA {
      return ShapedArray<Scalar>(shape: shape.dimensions, scalars: scalars)
    }
    return handle.makeHostCopy()
  }

  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public var scalars: [Scalar] {
    if handle.backend == .XLA {
      let (storage, _) = xlaTensor.fetchTensorValues(Scalar.self)
      return storage
    }
    return array.scalars
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: scalars)
  func _vjpScalars() -> (value: [Scalar], pullback: (Array<Scalar>.TangentVector) -> Tensor) {
    (
      value: scalars,
      pullback: { [shape = self.shape, device = self.device] v in
        Tensor(shape: shape, scalars: v.base, on: device)
      }
    )
  }
}

//===------------------------------------------------------------------------------------------===//
// Initialization
//===------------------------------------------------------------------------------------------===//

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
    return (Tensor(value, on: device), { $0.scalarized() })
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
    self.init([Scalar](vector), on: device)
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
    (
      value: Tensor(scalars, on: device),
      pullback: { v in
        Array<Scalar>.TangentVector(v.scalars)
      }
    )
  }

  @inlinable
  @derivative(of: init(shape:scalars:on:))
  static func _vjpInit(
    shape: TensorShape, scalars: [Scalar], on device: Device = .default
  ) -> (value: Tensor, pullback: (Tensor) -> Array<Scalar>.TangentVector) {
    (
      value: Tensor(shape: shape, scalars: scalars, on: device),
      pullback: { v in
        Array<Scalar>.TangentVector(v.scalars)
      }
    )
  }
}

// Background story on `TensorElementLiteral` and why it's necessary:
//
// Very importantly, we want users to be able to implicitly convert an array
// literal to a tensor. At first glance, a straightforward implementation would
// be conforming `Tensor` to `ExpressibleByArrayLiteral` with
// `ExpressibleBy(Float|Int|Bool)Literal` as a base case. However, it is not
// that simple. We have binary operators that take `(Tensor, Scalar)`, `(Scalar,
// Tensor)` as well as `(Tensor, Tensor)`. When `Tensor`s are convertible from
// both a scalar and an array literal, a scalar-tensor binary operator like `+`
// will not type check.
//
// One way to work around it is to define all tensor-tensor operators in a
// protocol extension, and all tensor-scalar and scalar-tensor operators on
// concrete `Tensor`. Protocol extensions are less favorable than concrete
// implementations, so the compiler will prefer the concrete implementation for
// a scalar-tensor operation. However, this would cause enormous code bloat and
// is entirely a hack.
//
// To resolve ambiguity, `Tensor` should not be expressible by scalar literal.
// There's already a lightweight syntax for converting a scalar to a tensor:
// `Tensor(x)`, so there is no strong need for implicit conversion. But we need
// to find a way to give `ExpressibleByArrayLiteral` a base case: what would the
// `ArrayLiteralElement` be if we want to support both `[1,2,3]` and `[[[1,2],
// [1,2]]]`? In the first case the array literal element is an integer, while
// in the second case the array literal itself should be a tensor. Based on this
// observation, we come up with an intermediate type: `TensorElementLiteral` as
// the `ArrayLiteralElement` of `Tensor`. By making `TensorElementLiteral`
// expressible by both array literal and scalar literal, `Tensor` can now be
// converted from an arbitrary-dimensional array literal.
//
// Due to protocol requirements, `TensorElementLiteral` has to be
// public. It is never supposed to be used directly by any user, so the library
// convention is to prepend an underscore to its name, making it
// `_TensorElementLiteral`.
//
// It would be nice to be able to remove this type when we can systematically
// resolve tensor-scalar/scalar-tensor op ambiguity someday, either through an
// improved `Expressible` model, or by introducing an attribute to tell the type
// checker which function to prefer when ambiguity occurs.

/// Represents a literal element for conversion to a `Tensor`.
///
/// - Note: Do not ever use this API directly. This is implicitly created
///   during the conversion from an array literal to a `Tensor`, and is purely
///   for implementation purposes.
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
    tensor = _Raw.pack(elements.map { $0.tensor })
  }
}

extension Tensor: ExpressibleByArrayLiteral {
  /// The type of the elements of an array literal.
  public typealias ArrayLiteralElement = _TensorElementLiteral<Scalar>

  /// Creates a tensor initialized with the given elements.
  /// - Note: This is for conversion from tensor element literals. This is a
  ///   separate method because `ShapedArray` initializers need to call it.
  @inlinable
  internal init(_tensorElementLiterals elements: [_TensorElementLiteral<Scalar>]) {
    self = _Raw.pack(elements.map { $0.tensor })
  }

  /// Creates a tensor initialized with the given elements.
  @inlinable
  public init(arrayLiteral elements: _TensorElementLiteral<Scalar>...) {
    precondition(!elements.isEmpty, "Cannot create a 'Tensor' with no elements.")
    self.init(_tensorElementLiterals: elements)
  }
}

//===------------------------------------------------------------------------------------------===//
// Equatable
//===------------------------------------------------------------------------------------------===//

extension Tensor: Equatable where Scalar: Equatable {
  @inlinable
  public static func == (lhs: Tensor, rhs: Tensor) -> Bool {
    guard lhs.shape == rhs.shape else {
      return false
    }
    return (lhs .== rhs).all()
  }

  @inlinable
  public static func != (lhs: Tensor, rhs: Tensor) -> Bool {
    guard lhs.shape == rhs.shape else {
      return true
    }
    return (lhs .!= rhs).any()
  }
}

//===------------------------------------------------------------------------------------------===//
// Description and Visualization
//===------------------------------------------------------------------------------------------===//

// String conversion.
extension Tensor: CustomStringConvertible {
  /// A textual representation of the tensor.
  ///
  /// - Note: use `fullDescription` for a non-pretty-printed description showing all scalars.
  public var description: String {
    @_semantics("autodiff.nonvarying")
    get {
      return array.description
    }
  }
}

extension Tensor {
  /// A textual representation of the tensor. Returns a summarized description if `summarize` is
  /// true and the element count exceeds twice the `edgeElementCount`.
  ///
  /// - Parameters:
  ///   - lineWidth: The max line width for printing. Used to determine number of scalars to print
  ///     per line.
  ///   - edgeElementCount: The maximum number of elements to print before and after summarization
  ///     via ellipses (`...`).
  ///   - summarizing: If true, summarize description if element count exceeds twice
  ///     `edgeElementCount`.
  public func description(
    lineWidth: Int = 80,
    edgeElementCount: Int = 3,
    summarizing: Bool = false
  ) -> String {
    return array.description(
      lineWidth: lineWidth,
      edgeElementCount: edgeElementCount,
      summarizing: summarizing)
  }

  /// A full, non-pretty-printed textual representation of the tensor, showing
  /// all scalars.
  public var fullDescription: String {
    @_semantics("autodiff.nonvarying")
    get {
      return array.fullDescription
    }
  }

  public var irText: String { XLATensor.irText(xlaTensor) }
}

// Xcode Playground display conversion.
extension Tensor: CustomPlaygroundDisplayConvertible {
  public var playgroundDescription: Any {
    @_semantics("autodiff.nonvarying")
    get {
      return description
    }
  }
}

// Mirror representation, used by debugger/REPL.
extension Tensor: CustomReflectable {
  public var customMirror: Mirror {
    @_semantics("autodiff.nonvarying")
    get {
      return Mirror(self, children: [], displayStyle: .struct)
    }
  }
}

//===------------------------------------------------------------------------------------------===//
// Codable Conformance
//===------------------------------------------------------------------------------------------===//

extension Tensor: Codable where Scalar: Codable {
  @inlinable
  public func encode(to encoder: Encoder) throws {
    var container = encoder.singleValueContainer()
    try container.encode(array)
  }

  @inlinable
  public init(from decoder: Decoder) throws {
    let container = try decoder.singleValueContainer()
    let array = try container.decode(ShapedArray<Scalar>.self)
    self.init(array)
  }
}

//===------------------------------------------------------------------------------------------===//
// Additive Group
//===------------------------------------------------------------------------------------------===//

extension Tensor: AdditiveArithmetic where Scalar: Numeric {
  /// The scalar zero tensor.
  public static var zero: Tensor {
    var zero = Tensor(0, on: _DeviceThreadLocalState.local.currentDevice)
    if _DeviceThreadLocalState.local.isReducedPrecision {
      zero = zero.toReducedPrecision
    }
    zero._isScalarZero = true
    return zero
  }

  /// Adds two tensors and produces their sum.
  /// - Note: `+` supports broadcasting.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public static func + (lhs: Tensor, rhs: Tensor) -> Tensor {
    if lhs._isScalarZero {
      return rhs
    } else if rhs._isScalarZero {
      return lhs
    }
    return _Raw.addV2(lhs, rhs)
  }

  /// Subtracts one tensor from another and produces their difference.
  /// - Note: `-` supports broadcasting.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public static func - (lhs: Tensor, rhs: Tensor) -> Tensor {
    if rhs._isScalarZero {
      return lhs
    }
    return _Raw.sub(lhs, rhs)
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: +)
  static func _vjpAdd(lhs: Tensor, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    (
      lhs + rhs,
      { [broadcastPb = BroadcastingPullback(lhs, rhs)] v in
        return broadcastPb(v, v)
      }
    )
  }

  @inlinable
  @derivative(of: -)
  static func _vjpSubtract(lhs: Tensor, rhs: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    (
      lhs - rhs,
      { [broadcastPb = BroadcastingPullback(lhs, rhs)] v in
        return broadcastPb(v, -v)
      }
    )
  }
}

//===------------------------------------------------------------------------------------------===//
// Multiplicative Group
//===------------------------------------------------------------------------------------------===//

extension Tensor: PointwiseMultiplicative where Scalar: Numeric {
  /// The scalar one tensor.
  @inlinable
  public static var one: Tensor { Tensor(1) }

  /// Returns the element-wise reciprocal of `self`.
  @inlinable
  public var reciprocal: Tensor { 1 / self }

  /// Multiplies two tensors element-wise and produces their product.
  /// - Note: `.*` supports broadcasting.
  public static func .* (lhs: Tensor, rhs: Tensor) -> Tensor {
    return lhs * rhs
  }
}

//===------------------------------------------------------------------------------------------===//
// Differentiable
//===------------------------------------------------------------------------------------------===//

extension Tensor: Differentiable & EuclideanDifferentiable where Scalar: TensorFlowFloatingPoint {
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
  init(repeating repeatedValue: Scalar, shape: TensorShape, on device: Device)
  var annotations: String { get }
  var shape: TensorShape { get }
  var summary: String { get }
}

public protocol DifferentiableTensorProtocol:
  TensorProtocol & Differentiable & EuclideanDifferentiable
where Scalar: TensorFlowFloatingPoint {
  @differentiable(reverse, wrt: self)
  func annotate(_ annotation: String) -> Self
}

extension Tensor: TensorProtocol {
  /// The annotations describing this tensor.
  public var annotations: String {
    switch handle.backend {
    case .XLA:
      return XLATensor.annotations(xlaTensor)
    case .TF_EAGER:
      return Device.defaultTFEager.annotationsAvailable
    }
  }

  /// An alias for annotations.
  public var summary: String { annotations }
}

extension Tensor: DifferentiableTensorProtocol
where Scalar: TensorFlowFloatingPoint {
  /// Adds an annotation.
  ///
  /// Note: Only X10 is supported. For other backends, umodified `self` is
  /// returned.
  ///
  /// - Parameter annotation: The annotation to be added.
  /// - Returns: The annotated tensor.
  @differentiable(reverse, wrt: self)
  public func annotate(_ annotation: String) -> Tensor<Scalar> {
    switch handle.backend {
    case .XLA:
      return Tensor<Scalar>(_xla: XLATensor.annotate(xlaTensor, annotation))
    case .TF_EAGER:
      return self
    }
  }

  @derivative(of: annotate)
  @usableFromInline
  func vjpAnnotate(_ annotation: String) -> (
    value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> Tensor<Scalar>
  ) {
    (annotate(annotation), { $0 })
  }
}
