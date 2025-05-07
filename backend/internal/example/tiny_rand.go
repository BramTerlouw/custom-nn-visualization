package example

import (
	"fmt"

	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/nnet/data"
	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/nnet/model"
	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/nnet/network"
)

// TrainRandMatrix
// Example usage of the neural network functions.
// Trains a network on a predefined random matrix dataset.
//
// Arguments:
//
//	None
//
// Returns:
//   - err:					(optional).
func TrainRandMatrix() error {
	const (
		inputSize    = 5
		outputSize   = 6
		learningRate = 0.01
		epochs       = 1
	)

	// Define random matrix dataset
	trainData := [][]float64{
		{0, 0.40, 0.33, 0.56, 0.66, 0.47},
		{1, 0.60, 0.77, 0.44, 0.12, 0.97},
		{2, 0.10, 0.37, 0.68, 0.22, 0.91},
		{3, 0.23, 0.86, 0.40, 0.19, 0.72},
		{4, 0.21, 0.94, 0.88, 0.37, 0.42},
		{5, 0.18, 0.26, 0.18, 0.43, 0.62},
	}

	hiddenLayerSizes := []int{10, 20, 10}
	layers := append([]int{inputSize}, hiddenLayerSizes...)
	layers = append(layers, outputSize)

	// Initialize neural network
	net := network.NewNetwork(layers, learningRate)

	// Preprocess training data
	processedTrain, trainTargets, err := data.Preprocess_Float64_Matrix(trainData, false, outputSize)
	if err != nil {
		return fmt.Errorf("failed to preprocess training data: %w", err)
	}

	// Train the model
	if err := model.TrainFeedForwardModel(net, processedTrain, trainTargets, epochs); err != nil {
		return fmt.Errorf("failed to train model: %w", err)
	}

	return nil
}
