package example

import (
	"fmt"

	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/nnet/data"
	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/nnet/model"
	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/nnet/network"
)

// TrainMnist
// Example usage of the neural network functions.
// Trains a network on MNIST csv dataset, and tests it
// with a test MNIST dataset.
//
// Arguments:
//
//	None
//
// Returns:
//   - err:					(optional).
func TrainMnist() error {
	const (
		inputSize     = 784
		outputSize    = 10
		learningRate  = 0.01
		epochs        = 1
		trainFilePath = "/Users/bramterlouw/Documents/custom-nn-visualization/mnist_train.csv"
		testFilePath  = "/Users/bramterlouw/Documents/custom-nn-visualization/mnist_test.csv"
	)

	hiddenLayerSizes := []int{100}
	layers := append([]int{inputSize}, hiddenLayerSizes...)
	layers = append(layers, outputSize)

	// Initialize neural network
	net := network.NewNetwork(layers, learningRate)

	// Preprocess training data
	trainData, trainTargets, err := data.Preprocess_CSV(trainFilePath, true, true, 0, outputSize)
	if err != nil {
		return fmt.Errorf("failed to preprocess training data: %w", err)
	}

	// Train the model
	if err := model.TrainFeedForwardModel(net, trainData, trainTargets, epochs); err != nil {
		return fmt.Errorf("failed to train model: %w", err)
	}

	// Preprocess test data
	testData, _, err := data.Preprocess_CSV(testFilePath, true, true, 0, outputSize)
	if err != nil {
		return fmt.Errorf("failed to preprocess test dataL %w", err)
	}

	// Evaluate the model
	if err := model.EvaluateFeedForwardModel(net, testData); err != nil {
		return fmt.Errorf("failed to evaluate model: %w", err)
	}

	return nil
}
