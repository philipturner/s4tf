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

import Swift

@usableFromInline
internal class TensorBuffer<Scalar> {
  /// Cached element count of the underlying buffer.
  let count: Int

  init(count: Int) { self.count = count }
}


@frozen
public struct ShapedArray<Scalar> {
  /// Contiguous memory storing scalars.
  internal var buffer: TensorBuffer<Scalar>

  /// The dimensions of the array.
  public private(set) var shape: [Int]

  /// Creates a `ShapedArray` from a `TensorBuffer` and a shape.
  internal init(buffer: __owned TensorBuffer<Scalar>, shape: __owned [Int]) {
    fatalError()

  }
}

extension ShapedArray {
  fileprivate mutating func ensureUniquelyReferenced() {
    fatalError()

//    }
  }
}

extension ShapedArray {
  /// The number of dimensions of the array.
  public var rank: Int {
    return shape.count
  }

  /// The total number of scalars in the array.
  public var scalarCount: Int {
    return buffer.count
  }

  /// Creates a `ShapedArray` with the same shape and scalars as the specified instance.
  public init(_ other: ShapedArray) {
    fatalError()

  }

  public init(shape: __owned [Int], scalars: __owned [Scalar]) {
    fatalError()
  }

  public init<S: Sequence>(shape: __owned [Int], scalars: __shared S) where S.Element == Scalar {
    fatalError()
  }

  /// Creates a `ShapedArray` from a scalar value.
  public init(_ scalar: __owned Scalar) {
    fatalError()
  }


  @inlinable
  @available(*, deprecated, renamed: "init(repeating:shape:)")
  public init(shape: __owned [Int], repeating repeatedValue: __owned Scalar) {
    self.init(repeating: repeatedValue, shape: shape)
  }

  public init(repeating repeatedValue: __owned Scalar, shape: __owned [Int]) {
    fatalError()
  }
}

extension ShapedArray: RandomAccessCollection, MutableCollection {
  public typealias Index = Int
  public typealias Element = ShapedArraySlice<Scalar>
  public typealias SubSequence = ShapedArraySlice<Scalar>

  public var indices: Range<Int> {
    return 0..<count
  }

  public var startIndex: Int {
    return 0
  }

  public var endIndex: Int {
    return count
  }

  public subscript(index: Int) -> Element {
    get {
      fatalError()
    }
    set {
      fatalError()

    }
  }

  public subscript(bounds: Range<Int>) -> SubSequence {
    get {
      fatalError()
    }
    set {
      fatalError()
    }
  }
}

extension ShapedArray {
  public func withUnsafeBufferPointer<Result>(
    _ body: (UnsafeBufferPointer<Scalar>) throws -> Result
  ) rethrows -> Result {
    fatalError()
//    return try buffer.withUnsafeBufferPointer { ptr in try body(ptr) }
  }

  public mutating func withUnsafeMutableBufferPointer<Result>(
    _ body: (inout UnsafeMutableBufferPointer<Scalar>) throws -> Result
  ) rethrows -> Result {
    fatalError()
//    ensureUniquelyReferenced()
//    return try buffer.withUnsafeMutableBufferPointer { ptr in try body(&ptr) }
  }
}

// Equatable conformance.
extension ShapedArray: Equatable where Scalar: Equatable {
  public static func == (lhs: ShapedArray, rhs: ShapedArray) -> Bool {
    fatalError()
  }
}

@frozen
public struct ShapedArraySlice<Scalar>  {
  @usableFromInline internal var base: ShapedArray<Scalar>
  @usableFromInline internal var baseIndices: [Int]
  @usableFromInline internal var bounds: Range<Int>?

  @inlinable
  internal init(
    base: __owned ShapedArray<Scalar>,
    baseIndices indices: __owned [Int] = [],
    bounds: Range<Int>? = nil
  ) {
    precondition(indices.count <= base.rank, "Number of base indices exceeds base rank.")
    precondition(
      zip(base.shape, indices).allSatisfy { $1 >= 0 && $1 < $0 },
      "Base indices are out of range")
    self.base = base
    self.baseIndices = indices
    self.bounds = bounds
  }
}

extension ShapedArraySlice {
  internal var indexingDepth: Int {
    return baseIndices.count
  }

  public var rank: Int {
    return base.rank - indexingDepth
  }

  public var shape: [Int] {
    if let bounds = bounds {
      return [bounds.count] + Array(base.shape.dropFirst(indexingDepth + 1))
    }
    return Array(base.shape.dropFirst(indexingDepth))
  }

  public var scalarCount: Int {
    return shape.reduce(1, *)
  }
}

extension ShapedArraySlice {
  public init(shape: __owned [Int], scalars: __owned [Scalar]) {
    self.init(base: ShapedArray(shape: shape, scalars: scalars))
  }

  public init<S: Sequence>(shape: __owned [Int], scalars: __shared S) where S.Element == Scalar {
    self.init(base: ShapedArray(shape: shape, scalars: scalars))
  }
  public init(_ scalar: __owned Scalar) {
    self.init(base: ShapedArray(scalar))
  }

  @inlinable
  @available(*, deprecated, renamed: "init(repeating:shape:)")
  public init(shape: __owned [Int], repeating repeatedValue: __owned Scalar) {
    self.init(repeating: repeatedValue, shape: shape)
  }
  
  public init(repeating repeatedValue: __owned Scalar, shape: __owned [Int]) {
    self.init(base: ShapedArray(repeating: repeatedValue, shape: shape))
  }
}


extension ShapedArraySlice: RandomAccessCollection, MutableCollection {
  public typealias Index = Int
  public typealias Element = ShapedArraySlice
  public typealias SubSequence = ShapedArraySlice

  public var indices: Range<Int> {
    fatalError()
  }

  public var startIndex: Int {
    return indices.startIndex
  }

  public var endIndex: Int {
    return indices.endIndex
  }

  public subscript(index: Int) -> Element {
    get {
      fatalError()
    }
    set {
      fatalError()
    }
  }

  public subscript(bounds: Range<Int>) -> SubSequence {
    get {
      fatalError()
    }
    set {
      fatalError()
    }
  }
}

extension ShapedArraySlice: Equatable where Scalar: Equatable {
  public static func == (lhs: ShapedArraySlice, rhs: ShapedArraySlice) -> Bool {
    fatalError()
  }
}

