// Copyright 2020 TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

@_implementationOnly import x10_xla_tensor_tf_ops
@_implementationOnly import x10_xla_tensor_wrapper

/// Type-erased tensor type on which the fundamental operators are implemented.
struct XLATensor {
  init(_handle: UnsafeMutablePointer<OpaqueXLATensor>) {
    handleDeleter = Handle(_handle: _handle)
  }

  init(_ handle: Handle) {
    handleDeleter = handle
  }

  init?(_ handle: _AnyTensorHandle) {
    if let handle = handle as? Handle {
      self.init(handle)
    } else {
      return nil
    }
  }

  /// The device on which `self` is allocated.
  public var device: Device {
    defer { _fixLifetime(self) }
    return XLATensor_device(handle).device
  }

  var handle: UnsafeMutablePointer<OpaqueXLATensor> {
    return handleDeleter.handle
  }

  // Implementation detail for deleting the pointer.
  class Handle: _AnyTensorHandle {
    init(_handle: UnsafeMutablePointer<OpaqueXLATensor>) {
      handle = _handle
    }

    deinit { destroyTensor(handle) }

    let handle: UnsafeMutablePointer<OpaqueXLATensor>
    var xlaTensor: XLATensor { XLATensor(self) }

    var _tfeTensorHandle: TFETensorHandle { fatalError("Not a tf handle") }
    var rank: Int { xlaTensor.shape.count }
    var shape: TensorShape { TensorShape(xlaTensor.shape) }

    public var backend: Device.Backend { .XLA }
  }

  var tensorHandle: _AnyTensorHandle { handleDeleter }

  let handleDeleter: Handle
}

extension Tensor {
  init(_xla: XLATensor) {
    precondition(
      _xla.dtype == Scalar.xlaTensorScalarType,
      "Type mismatch constructing from XLATensor:"
        + "\(_xla.dtype) vs \(Scalar.xlaTensorScalarType)")
    handle = TensorHandle(handle: _xla.tensorHandle)
  }

  init(_xlaHandle: UnsafeMutablePointer<OpaqueXLATensor>) {
    self.init(_xla: XLATensor(_handle: _xlaHandle))
  }

  var xlaHandle: UnsafeMutablePointer<OpaqueXLATensor> { return xlaTensor.handle }

  var xlaTensor: XLATensor {
    guard let xlaTensor = XLATensor(handle.handle) else {
      fatalError("Must be an XLATensor to convert to XlaTensor")
    }
    return xlaTensor
  }
}

extension XLATensor {
  /// TODO(parkers): Add support for other types and aliasing.
  static func make<Scalar: XLAScalarType>(
    _ data: [Scalar], _ dims: [Int], on device: Device = Device.default
  ) -> XLATensor {
    data.withUnsafeBufferPointer { data in return make(data, dims, on: device) }
  }

  static func make<Scalar: XLAScalarType>(_ data: Scalar, on device: Device = Device.default)
    -> XLATensor
  {
    return XLATensor(
      _handle: XLATensor_makeScalar(data.xlaScalar, Scalar.xlaTensorScalarType, device.cdevice))
  }

  static func make<Scalar: XLAScalarType>(
    _ data: UnsafeBufferPointer<Scalar>, _ dims: [Int], on device: Device = Device.default
  )
    -> XLATensor
  {
    dims.withUnsafeBufferPointer { dims in
      return XLATensor(
        _handle:
          copyTensor(
            Scalar.xlaTensorScalarType, data.baseAddress, data.count, dims.baseAddress, dims.count,
            device.cdevice
          ))
    }
  }

  static func make<Scalar: XLAScalarType>(
    _ data: [Scalar], _ dims: [Int], toReducedPrecision: Bool,
    directlyOn device: Device = Device.default
  ) -> XLATensor {
    data.withUnsafeBufferPointer { data in
      return make(data, dims, toReducedPrecision: toReducedPrecision, directlyOn: device)
    }
  }

  static func make<Scalar: XLAScalarType>(
    _ data: UnsafeBufferPointer<Scalar>, _ dims: [Int], toReducedPrecision: Bool,
    directlyOn device: Device = Device.default
  )
    -> XLATensor
  {
    dims.withUnsafeBufferPointer { dims in
      return XLATensor(
        _handle:
          copyTensorAndMakeResident(
            Scalar.xlaTensorScalarType, data.baseAddress, data.count, dims.baseAddress, dims.count,
            device.cdevice, toReducedPrecision
          ))
    }
  }

  var shape: [Int] {
    defer { _fixLifetime(self) }
    let shape = fetchTensorShape(handle)!
    let rank = XLAShape_getRank(shape)
    let data = XLAShape_getDimensions(shape)
    let result = Array(UnsafeBufferPointer(start: data!, count: rank))
    destroyXLAShape(shape)
    return result.map { Int($0) }
  }

  func fetchTensorValues<Scalar: XLAScalarType>(_ t: Scalar.Type) -> (data: [Scalar], dims: [Int]) {
    defer { _fixLifetime(self) }
    let materialized = XLATensor_materialize(handle)!
    let dims = shape
    let count = shape.reduce(1, *)
    precondition(
      MaterializedTensor_getType(materialized) == Scalar.xlaTensorScalarType,
      "Types mismatch when fetching tensor values.")
    let data = Array(
      UnsafeBufferPointer(
        start:
          UnsafePointer<Scalar>(OpaquePointer(MaterializedTensor_getData(materialized))),
        count: count))
    destroyMaterializedTensor(materialized)
    return (data: data, dims: dims)
  }

  var dtype: XLATensorScalarType {
    defer { _fixLifetime(self) }
    return XLATensor_dtype(handle)
  }
  var physicalScalarType: XLATensorScalarType {
    defer { _fixLifetime(self) }
    return XLATensor_physical_scalar_type(handle)
  }
}

extension Array where Element == Int64 {
  func withArrayRef<Result>(_ body: (Int64ArrayRef) throws -> Result) rethrows -> Result {
    return try withUnsafeBufferPointer { buf in
      return try body(Int64ArrayRef(data: buf.baseAddress, size: buf.count))
    }
  }
}

extension Array where Element == XLATensor {
  func withArrayRef<Result>(_ body: (OpaqueXLATensorArrayRef) throws -> Result) rethrows -> Result {
    defer { _fixLifetime(self) }
    return try map { $0.handle }.withUnsafeBufferPointer { buf in
      return try body(OpaqueXLATensorArrayRef(data: buf.baseAddress, size: buf.count))
    }
  }
}

extension Array where Element: AnyTensor {
  func withArrayRef<T, Result>(_ body: (OpaqueXLATensorArrayRef) throws -> Result) rethrows
    -> Result
  where Element == Tensor<T> {
    defer { _fixLifetime(self) }
    return try map { $0.xlaHandle }.withUnsafeBufferPointer { buf in
      return try body(OpaqueXLATensorArrayRef(data: buf.baseAddress, size: buf.count))
    }
  }
}

extension Array where Element == PaddingConfigDimension {
  func withArrayRef<Result>(_ body: (inout PaddingConfig) -> Result) -> Result {
    defer { _fixLifetime(self) }
    return withUnsafeBufferPointer {
      (_ dimensions: UnsafeBufferPointer<PaddingConfigDimension>) -> Result in
      var paddingConfig = PaddingConfig(dimensions: dimensions.baseAddress, count: count)
      return body(&paddingConfig)
    }
  }
}

extension Optional where Wrapped == XLAScalarType.Type {
  var xlaOptionalType: Optional_XLAScalarType {
    defer { _fixLifetime(self) }
    if let type = self {
      return Optional_XLAScalarType(has_value: true, type: type.xlaTensorScalarType)
    }
    return Optional_XLAScalarType(has_value: false, type: XLATensorScalarType(rawValue: 0))
  }
}

extension Tensor {
  public var xlaIrText: String {
    let str = XLATensor_xla_ir_text(xlaTensor.handle)
    defer { DeleteString(str) }
    return String(cString: GetStringCStr(str))
  }
  var placeholder: Tensor {
    return Tensor(_xlaHandle: XLATensor_makePlaceholder(self.xlaHandle, 0))
  }
}

extension Array where Element == AnyTensor {
  func withArrayRef<Result>(_ body: (OpaqueXLATensorArrayRef) throws -> Result) rethrows -> Result {
    try self.map { $0.scalarType.unwrapTensor($0) }.withArrayRef { try body($0) }
  }
}

extension TensorFlowScalar {
  static func unwrapTensor(_ t: AnyTensor) -> XLATensor {
    return (t as! Tensor<Self>).xlaTensor
  }
  static func wrapTensor(_ t: XLATensor) -> AnyTensor {
    return Tensor<Self>(_xla: t)
  }
  static func makePlaceholder(_ t: AnyTensor, i: Int = 0) -> AnyTensor {
    return Tensor<Self>(
      _xlaHandle: XLATensor_makePlaceholder((t as! Tensor<Self>).xlaHandle, Int32(i)))
  }
}
