package model

import (
	"fmt"

	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/nnet/network"
)

// TrainFeedForwardModel
// Performs the Train function in a loop for x epochs.
//
// Arguments:
//   - net:				neural network to train.
//   - processed_train:	processed train data.
//   - targets_train:	training target values.
//   - epochs:			amount of train epochs.
//
// Returns:
//   - err: 			(optional).
func TrainFeedForwardModel(net *network.Network, processed_train, targets_train [][]float64, epochs int) error {

	for epoch := range epochs {
		fmt.Printf("Epoch %d\n", epoch+1)

		for i := range processed_train {
			net.Train(processed_train[i][1:], targets_train[i])
		}

	}
	fmt.Println("Training completed")
	return nil
}
