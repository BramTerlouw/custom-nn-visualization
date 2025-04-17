package neuralnetwork

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type Network struct {
	inputSize, hiddenSize, outputSize int
	WeightsHidden, WeightsOutput      *mat.Dense
	learningRate                      float64
}

func NewNetwork(inputSize, hiddenSize, outputSize int, learningRate float64) *Network {

	// Initialize network
	n := &Network{
		inputSize:    inputSize,
		hiddenSize:   hiddenSize,
		outputSize:   outputSize,
		learningRate: learningRate,
	}

	// Initialize weights with random values
	n.WeightsHidden = mat.NewDense(hiddenSize, inputSize, nil)
	n.WeightsOutput = mat.NewDense(outputSize, hiddenSize, nil)

	// Assign random float64 values to each index in the hidden weights matrix
	for i := 0; i < hiddenSize*inputSize; i++ {
		n.WeightsHidden.RawMatrix().Data[i] = rand.NormFloat64()
	}

	// Assign random float64 values to each index in the output weights matrix
	for i := 0; i < outputSize*hiddenSize; i++ {
		n.WeightsOutput.RawMatrix().Data[i] = rand.NormFloat64()
	}

	return n
}

func (net *Network) Train(inputData []float64, targetData []float64) {
	// forward propagation
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := Matrix_multiply(net.WeightsHidden, inputs)
	hiddenOutputs := Apply_fn(Sigmoid, hiddenInputs)
	finalInputs := Matrix_multiply(net.WeightsOutput, hiddenOutputs)
	finalOutputs := Apply_fn(Sigmoid, finalInputs)

	// find errors
	targets := mat.NewDense(len(targetData), 1, targetData)
	outputErrors := Subtract(targets, finalOutputs)
	hiddenErrors := Matrix_multiply(net.WeightsOutput.T(), outputErrors)

	// backpropagate
	net.WeightsOutput = Add(net.WeightsOutput,
		Scale(net.learningRate,
			Matrix_multiply(Element_multiply(outputErrors, SigmoidPrime(finalOutputs)),
				hiddenOutputs.T()))).(*mat.Dense)

	net.WeightsHidden = Add(net.WeightsHidden,
		Scale(net.learningRate,
			Matrix_multiply(Element_multiply(hiddenErrors, SigmoidPrime(hiddenOutputs)),
				inputs.T()))).(*mat.Dense)
}

func addScalar(i float64, m mat.Matrix) mat.Matrix {
	rows, cols := m.Dims()
	a := make([]float64, rows*cols)
	for x := 0; x < rows*cols; x++ {
		a[x] = i
	}
	n := mat.NewDense(rows, cols, a)
	return Add(m, n)
}
