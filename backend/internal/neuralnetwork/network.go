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
			data[j] = randomFloat(float64(layers[0]))
		}

		// Create the weights matrix for layer.
		weights[layerIndex] = mat.NewDense(rows, cols, data)
	}

	// Create the neural network object with layers, generated weigths and
	// learningRate.
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

	// Return the output values of the current node and all values of the activated
	// layers.
	return current, activations
}

func (net *Network) Train(inputData []float64, targetData []float64) {

	// Execute a forward pass through whole network.
	outputs, activations := net.Forward(inputData)

	// Convert the validation values into 1dim matrix where all values are put
	// into rows in single column.
	targets := mat.NewDense(len(targetData), 1, targetData)

	// Calculate error by subtracting output values (outputs matrix) from forward
	// pass from the validation values (targets matrix).
	outputErrors := Subtract_matrix(targets, outputs)

	// Iterate through all the layers and perform backpropagation, starting with the
	//  weights between the output layer and the last hidden layer.
	errors := outputErrors
	for i := len(net.Weights) - 1; i >= 0; i-- {

		// layerInput of current layer is the output of previous node (len(net.Weights)
		//  - 1). layerOutput is the output of the current layer.
		layerInput := activations[i]
		layerOutput := activations[i+1]

		// Calculate the change (delta) of the weights by multiplying each error in the
		// output matrix with the corresponding derivative (how sensitive to change) of
		// each output value in the output matrix of current layer.
		delta := Element_multiply(errors, SigmoidPrime(layerOutput))

		// Calculate the value used to update weights between current layer and current
		// layer - 1 by multiplying the learning rate * (weight difference * input of
		// the previous layer).
		weightUpdate := Scale_matrix(net.learningRate, Matrix_multiply(delta, layerInput.T()))

		// Update the weights with the update value.
		net.Weights[i] = Add_matrix(net.Weights[i], weightUpdate).(*mat.Dense)

		// Calculate errors previous layer, except for the input layer.
		if i > 0 {
			errors = Matrix_multiply(net.Weights[i].T(), delta)
		}
	}
}
