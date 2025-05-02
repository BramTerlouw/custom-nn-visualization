package example

import (
	"fmt"

	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/nnet/data"
	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/nnet/model"
	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/nnet/network"
)

func TrainMnist() {

	// ARGS/ENVS
	INPUT_SIZE := 784
	HIDDEN_LAYER_SIZES := []int{100}
	OUTPUT_SIZE := 10

	layers := append([]int{INPUT_SIZE}, HIDDEN_LAYER_SIZES...)
	layers = append(layers, OUTPUT_SIZE)

	// Create a network for MNIST: 784 inputs, 100 hidden, 10 outputs
	net := network.NewNetwork(layers, 0.01)

	// Preproces train
	processed_train, targets_train, err1 := data.Preprocess_CSV("/Users/bramterlouw/Documents/custom-nn-visualization/mnist_train.csv", true, true, 0, net.Layers[len(net.Layers)-1])
	if err1 != nil {
		fmt.Print(err1)
	}
	fmt.Println("Preprocessing train data completed")

	err2 := model.TrainFeedForwardModel(net, processed_train, targets_train, 1)
	if err2 != nil {
		fmt.Print(err2)
	}

	processed_test, _, err3 := data.Preprocess_CSV("/Users/bramterlouw/Documents/custom-nn-visualization/mnist_test.csv", true, true, 0, net.Layers[len(net.Layers)-1])
	if err3 != nil {
		fmt.Print(err3)
	}
	fmt.Println("Preprocessing test data completed")

	err4 := model.EvaluateFeedForwardModel(net, processed_test)
	if err4 != nil {
		fmt.Print(err4)
	}
}
