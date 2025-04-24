package model

import (
	"fmt"
	"time"

	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/nnet/data"
	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/nnet/network"
	"gonum.org/v1/gonum/mat"
)

func TrainFeedForwardModel(
	net *network.Network,
	train, test string,
	epochs int,
) error {

	// Preproces train
	t1 := time.Now()
	processed_train, targets_train, err := data.Preprocess(train, true, true, 0, net.Layers[len(net.Layers)-1])
	if err != nil {
		return fmt.Errorf("preprocess train data failed: %w", err)
	}

	fmt.Println("Preprocessing train data completed")
	elapsed1 := time.Since(t1)
	fmt.Printf("Time taken to preprocess: %s\n", elapsed1)
	// End preproces train

	// Train model
	t2 := time.Now()
	for epoch := 0; epoch < epochs; epoch++ {
		fmt.Printf("Epoch %d\n", epoch+1)

		for i := 0; i < len(processed_train); i++ {

			net.Train(processed_train[i][1:], targets_train[i])
		}

	}

	fmt.Println("Training completed")
	elapsed2 := time.Since(t2)
	fmt.Printf("Time taken to train: %s\n", elapsed2)
	// End train model

	// Preproces test
	t3 := time.Now()
	processed_test, _, err := data.Preprocess(test, true, true, 0, net.Layers[len(net.Layers)-1])
	if err != nil {
		return fmt.Errorf("preprocess train data failed: %w", err)
	}

	fmt.Println("Preprocessing test data completed")
	elapsed3 := time.Since(t3)
	fmt.Printf("Time taken to preprocess: %s\n", elapsed3)
	// End preproces test

	// Evaluate model
	t4 := time.Now()
	var score = 0
	for i := 0; i < len(processed_test)-1; i++ {

		bestIdx, _ := predict(net, processed_test[i][1:])

		if bestIdx == int(processed_test[i][0]) {
			score++
		}
	}

	fmt.Println("Evaluation completed")
	elapsed4 := time.Since(t4)
	fmt.Printf("Time taken to evaluate: %s\n", elapsed4)
	fmt.Printf("Score: %d\n", score)

	return nil
}

func predict(net *network.Network, inputs []float64) (int, mat.Matrix) {

	// Perform forward propagation to get the network's output.
	outputs, _ := net.Forward(inputs)

	best := 0
	highest := 0.0

	// Determine the predicted class by finding the output neuron
	// with the highest value.
	for i := 0; i < net.Layers[len(net.Layers)-1]; i++ {
		if outputs.At(i, 0) > highest {
			best = i
			highest = outputs.At(i, 0)
		}
	}

	return best, outputs
}
