package network

import (
	"gonum.org/v1/gonum/mat"
)

type Network struct {
	Layers       []int        // Integer array where each index represents a layer where the amount of neurons equals the value on index.
	Weights      []*mat.Dense // Array of matrices where each index represents the weights of a single layer.
	learningRate float64      // Float64 value representing the learningrate of the neural network.
}

// NewNetwork
// Constructor function for initializing a new neural
// network struct.
//
// Arguments:
//   - layer:			array, each entry a layer size.
//   - learningRate:	float64 value for learning rate adjustment.
//
// Returns:
//   - *Network:		neural network struct.
func NewNetwork(layers []int, learningRate float64) *Network {

	// Initialize array of matrices equal to nr of layers - output layer.
	weights := make([]*mat.Dense, len(layers)-1)

	for layerIdx := range len(layers) - 1 {

		// Weight dims calculated with nr of nodes current layer
		// * nr of nodes in next layer (layerIndex+1).
		rows, cols := layers[layerIdx+1], layers[layerIdx]
		data := make([]float64, rows*cols)

		// Asign values and create weights matrix for layer.
		for j := range data {
			data[j] = randomFloat(float64(layers[0]))
		}
		weights[layerIdx] = mat.NewDense(rows, cols, data)
	}

	return &Network{
		Layers:       layers,
		Weights:      weights,
		learningRate: learningRate,
	}
}

// Forward
// Network function used to peform one forward pass through
// the network. Iterates through the matrix of weights,
// calculates inputs and apply's sigmoid activation function.
//
// Arguments:
//   - inputData:			1 input value (float64 array).
//
// Returns:
//   - current:				layer output (last layer).
//   - activations:			matrix with all output values (activations).
func (net *Network) Forward(inputData []float64) (mat.Matrix, []mat.Matrix) {

	// Convert input array -> 1dim matrix where all values are put into rows
	// in single col -> first layer into the activation array of layers.
	inputs := mat.NewDense(len(inputData), 1, inputData)
	activations := []mat.Matrix{inputs}

	// Set inputs as current layer and start interating over layers of weights.
	current := inputs
	for _, weights := range net.Weights {

		// Calcute input of the next neuron by multiplying current * weights
		// of connection between current and next neuron.
		neuronInput := Matrix_multiply(weights, current)

		// Apply sigmoid activation function on calculated inputs of neuron
		// and set neuron as current neuron.
		current = Apply_fn(Sigmoid, neuronInput).(*mat.Dense)

		// Add the activation of this (hidden) neuron to activations array.
		activations = append(activations, current)
	}

	// Return output values of current node and all values of activated layers.
	return current, activations
}

// Train
// Network function used to train the network by performing
// backpropagation (adjust weights based on error) after one
// forward pass.
//
// Arguments:
//   - inputData:			1 input value (float64 array).
//   - targetData			OneHotEncoded array with target.
//
// Returns:
//   - n.v.t.
func (net *Network) Train(inputData, targetData []float64) {

	// Execute forward pass through network.
	outputs, activations := net.Forward(inputData)

	// Convert validation values into 1dim matrix where all values
	// are put into rows in single column.
	targets := mat.NewDense(len(targetData), 1, targetData)

	// Calculate error by subtracting output values (outputs matrix)
	// from forward pass from validation values (targets matrix).
	outputErrors := Subtract_matrix(targets, outputs)

	// Iterate through all layers and perform backpropagation, starting
	//  with weights between output layer and last hidden layer.
	errors := outputErrors
	for i := len(net.Weights) - 1; i >= 0; i-- {

		// layerInput of current layer is output of previous node
		// (len(net.Weights) - 1). layerOutput is output of current layer.
		layerInput := activations[i]
		layerOutput := activations[i+1]

		// Calculate change (delta) of weights by multiplying each error in
		// output matrix with corresponding derivative (how sensitive to change) of
		// each output value in output matrix of current layer.
		delta := Element_multiply(errors, SigmoidPrime(layerOutput))

		// Calculate value used to update weights between current layer previous
		//  by multiplying learning rate * (weight difference * input of previous layer).
		weightUpdate := Scale_matrix(net.learningRate, Matrix_multiply(delta, layerInput.T()))

		// Update weights with update value.
		net.Weights[i] = Add_matrix(net.Weights[i], weightUpdate).(*mat.Dense)

		// Calculate errors previous layer, except for input layer.
		if i > 0 {
			errors = Matrix_multiply(net.Weights[i].T(), delta)
		}
	}
}
