package model

import (
	"fmt"

	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/nnet/network"
)

// EvaluateFeedForwardModel
// Use the trained network with test data and calculate
// the accuracu of the network.
//
// Arguments:
//   - net:				neural network to train.
//   - processed_test:	processed test data.
//
// Returns:
//   - err: 			(optional).
func EvaluateFeedForwardModel(net *network.Network, processed_test [][]float64) error {

	var score = 0
	for i := range len(processed_test) - 1 {
		outputs, _ := net.Forward(processed_test[i][1:])

		bestIdx := 0
		highest := 0.0

		for i := range net.Layers[len(net.Layers)-1] {
			if outputs.At(i, 0) > highest {
				bestIdx = i
				highest = outputs.At(i, 0)
			}
		}

		if bestIdx == int(processed_test[i][0]) {
			score++
		}
	}

	percentage := calcPerc(score, len(processed_test))
	fmt.Println("Evaluation completed")
	fmt.Printf("Score: %.2f%%\n", percentage)

	return nil
}

// calcPerc
// Calculate the percentage of the total.
//
// Arguments:
//   - score:			amount of good predictions.
//   - total:			amount of total predictions.
//
// Returns:
//   - output: 			calculated percentage.
func calcPerc(score, total int) float64 {
	return (float64(score) / float64(total) * 100)
}
