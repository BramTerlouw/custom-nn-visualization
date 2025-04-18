package neuralnetwork

import (
	"gonum.org/v1/gonum/mat"
)

type Network struct {
	Layers       []int        // Integer array where each index represents a layer where the amount of neurons equals the value on index.
	Weights      []*mat.Dense // Array of matrices where each index represents the weights of a single layer.
	learningRate float64      // Float64 value representing the learningrate of the neural network.
}

func NewNetwork(layers []int, learningRate float64) *Network {

	// Initialize array of matrices equal to the amount of layers minus one.
	weights := make([]*mat.Dense, len(layers)-1)

	// Loop over all layers minus one.
	for layerIndex := 0; layerIndex < len(layers)-1; layerIndex++ {

		// Weight dims are calculated with amount of nodes in current layer
		// and the next layer (layerIndex+1).
		rows, cols := layers[layerIndex+1], layers[layerIndex]
		data := make([]float64, rows*cols)

		// Asign random distributed float value to all data values.
		for j := range data {
			data[j] = randomArray(float64(layers[0]))
		}

		// Create the weights matrix for layer.
		weights[layerIndex] = mat.NewDense(rows, cols, data)
	}

	// Create the neural network object with layers, generated weigths and learningRate.
	return &Network{
		Layers:       layers,
		Weights:      weights,
		learningRate: learningRate,
	}
}

func (net *Network) Forward(inputData []float64) (mat.Matrix, []mat.Matrix) {

	// Convert input array into 1dim matrix where all values are put into rows
	// in single col. Add these as the first layer into the activation array of
	// layer matrices.
	inputs := mat.NewDense(len(inputData), 1, inputData)
	activations := []mat.Matrix{inputs}

	// Set inputs as the current layer and start interating over layers of weights.
	current := inputs
	for _, weights := range net.Weights {

		// Calcute the input of the next neuron by multiplying current with weights
		// of connection between current neuron and next neuron.
		neuronInput := Matrix_multiply(weights, current)

		// Apply sigmoid activation function on the calculated inputs of the neuron
		// and set the neuron as the current neuron.
		current = Apply_fn(Sigmoid, neuronInput).(*mat.Dense)

		// Add the activation of this (hidden) neuron activations array of matrices.
		activations = append(activations, current)
	}

	// Return the output values of the current node and all values of the activated layers.
	return current, activations
}
