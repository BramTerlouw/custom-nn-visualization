package example

import (
	"fmt"

	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/nnet/data"
	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/nnet/model"
	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/nnet/network"
)

func TrainRandMatrix() {

	train_data := [][]float64{
		{0, 0.40, 0.33, 0.56, 0.66, 0.47},
		{1, 0.60, 0.77, 0.44, 0.12, 0.97},
		{2, 0.10, 0.37, 0.68, 0.22, 0.91},
		{3, 0.23, 0.86, 0.40, 0.19, 0.72},
		{4, 0.21, 0.94, 0.88, 0.37, 0.42},
		{5, 0.18, 0.26, 0.18, 0.43, 0.62},
	}

	// ARGS/ENVS
	INPUT_SIZE := 5
	HIDDEN_LAYER_SIZES := []int{10, 20, 10}
	OUTPUT_SIZE := 6

	layers := append([]int{INPUT_SIZE}, HIDDEN_LAYER_SIZES...)
	layers = append(layers, OUTPUT_SIZE)

	// Create a network for MNIST: 784 inputs, 100 hidden, 10 outputs
	net := network.NewNetwork(layers, 0.01)

	processed_train, targets_train, err1 := data.Preprocess_Float64_Matrix(train_data, false, net.Layers[len(net.Layers)-1])
	if err1 != nil {
		fmt.Print(err1)
	}
	fmt.Println("Preprocessing train data completed")

	// Train the network
	err := model.TrainFeedForwardModel(net, processed_train, targets_train, 1)
	if err != nil {
		fmt.Print(err)
	}
}
