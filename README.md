# Swift for TensorFlow Deep Learning Library

Get a taste of *protocol-oriented differentiable programming*.

This repository hosts [Swift for TensorFlow][s4tf]'s deep learning library,
available both as a part of Swift for TensorFlow toolchains and as a Swift
package.

## Usage

This library is being [automatically integrated][integrated] in Swift for
TensorFlow toolchains. You do not need to add this library as a Swift Package
Manager dependency.

### Use Google Colaboratory

[**Open an empty Colaboratory now**][blank_colab] to try out Swift,
TensorFlow, differentiable programming, and deep learning.

> For detailed usage and troubleshooting, see [Usage][usage] on the Swift for
TensorFlow project homepage.

#### Define a model

Simply import `TensorFlow` to get the full power of TensorFlow.

```swift
import TensorFlow

let hiddenSize: Int = 10

struct Model: Layer {
    var layer1 = Dense<Float>(inputSize: 4, outputSize: hiddenSize, activation: relu)
    var layer2 = Dense<Float>(inputSize: hiddenSize, outputSize: hiddenSize, activation: relu)
    var layer3 = Dense<Float>(inputSize: hiddenSize, outputSize: 3, activation: identity)
    
    @differentiable(reverse)
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: layer1, layer2, layer3)
    }
}
```

#### Initialize a model and an optimizer

```swift
var classifier = Model()
let optimizer = SGD(for: classifier, learningRate: 0.02)
Context.local.learningPhase = .training
// Dummy data.
let x: Tensor<Float> = Tensor(randomNormal: [100, 4])
let y: Tensor<Int32> = Tensor(randomUniform: [100])
```

#### Run a training loop

One way to define a training epoch is to use the
[`gradient(at:in:)`][gradient] function.

```swift
for _ in 0..<1000 {
    let 𝛁model = gradient(at: classifier) { classifier -> Tensor<Float> in
        let ŷ = classifier(x)
        let loss = softmaxCrossEntropy(logits: ŷ, labels: y)
        print("Loss: \(loss)")
        return loss
    }
    optimizer.update(&classifier, along: 𝛁model)
}
```

Another way is to make use of methods on `Differentiable` or `Layer` that
produce a backpropagation function. This allows you to compose your derivative
computation with great flexibility.

```swift
for _ in 0..<1000 {
    let (ŷ, backprop) = classifier.appliedForBackpropagation(to: x)
    let (loss, 𝛁ŷ) = valueWithGradient(at: ŷ) { ŷ in softmaxCrossEntropy(logits: ŷ, labels: y) }
    print("Model output: \(ŷ), Loss: \(loss)")
    let (𝛁model, _) = backprop(𝛁ŷ)
    optimizer.update(&classifier, along: 𝛁model)
}
```

For more models, go to [**tensorflow/swift-models**][swift-models].

## Development

Documentation covering development can be found in the [Developer Guide](Documentation/Development.md).

## Bugs

Please report bugs and feature requests using GitHub issues in this repository.

## Community

Discussion about Swift for TensorFlow happens on the
[swift@tensorflow.org][forum]
mailing list.

## Contributing

We welcome contributions: please read the [Contributor Guide](CONTRIBUTING.md)
to get started. It's always a good idea to discuss your plans on the mailing
list before making any major submissions.

## Code of Conduct

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to making participation in our project and
our community a harassment-free experience for everyone, regardless of age, body
size, disability, ethnicity, gender identity and expression, level of
experience, education, socio-economic status, nationality, personal appearance,
race, religion, or sexual identity and orientation.

The Swift for TensorFlow community is guided by our [Code of
Conduct](CODE_OF_CONDUCT.md), which we encourage everybody to read before
participating.

[s4tf]: https://github.com/tensorflow/swift
[integrated]: https://github.com/apple/swift/tree/tensorflow#customize-tensorflow-support
[blank_colab]: https://colab.research.google.com/notebook#create=true&language=swift
[usage]: https://github.com/tensorflow/swift/blob/main/Usage.md
[gradient]: https://www.tensorflow.org/swift/api_docs/Functions#/s:10TensorFlow8gradient2at2in13TangentVectorQzx_AA0A0Vyq_GxXEtAA14DifferentiableRzAA0aB13FloatingPointR_r0_lF
[swift-models]: https://github.com/tensorflow/swift-models
[toolchain]: https://github.com/tensorflow/swift/blob/main/Installation.md
[forum]: https://groups.google.com/a/tensorflow.org/d/forum/swift
