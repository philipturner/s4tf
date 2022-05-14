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

@_exported import _Differentiation

extension Array.DifferentiableView:
  BidirectionalCollection,
  Collection,
  MutableCollection,
  RandomAccessCollection,
  RangeReplaceableCollection,
  Sequence
where Element: Differentiable {
  public typealias Element = Array<Element>.Element
  public typealias Index = Array<Element>.Index
  public typealias Indices = Array<Element>.Indices
  public typealias SubSequence = Array<Element>.SubSequence

  @inlinable
  public subscript(position: Array<Element>.Index) -> Element {
    _read { yield base[position] }
    set { base[position] = newValue }
  }

  @inlinable
  public subscript(bounds: Range<Array<Element>.Index>) -> Self.SubSequence {
    _read { yield base[bounds] }
    set { base[bounds] = newValue }
  }

  @inlinable
  public mutating func replaceSubrange<C>(_ subrange: Range<Self.Index>, with newElements: C) where C : Collection, Self.Element == C.Element {
    fatalError("withUnsafeBufferPointer unimplemented because TensorBuffer is abstract")
  }

  @inlinable
  public var startIndex: Index { base.startIndex }

  @inlinable
  public var endIndex: Index { base.endIndex }

  @inlinable
  public init() { self.init(.init()) }
}
