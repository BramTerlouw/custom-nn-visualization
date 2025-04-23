package example

import (
	"fmt"
	"time"

	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/nnet/network"
)

func TinyRandTrain(net *network.Network) error {

	// Start timing the evaluation process.
	t1 := time.Now()

	// Define the training data set
	train_data := [][]float64{
		{0, 0.40, 0.33, 0.56, 0.66, 0.47},
		{1, 0.60, 0.77, 0.44, 0.12, 0.97},
		{2, 0.10, 0.37, 0.68, 0.22, 0.91},
		{3, 0.23, 0.86, 0.40, 0.19, 0.72},
		{4, 0.21, 0.94, 0.88, 0.37, 0.42},
		{5, 0.18, 0.26, 0.18, 0.43, 0.62},
	}

	// Train the network for a fixed number of epochs (x iterations over the
	// dataset).
	for epochs := 0; epochs < 5; epochs++ {

		// Process each array of float64 input values.
		for i := range train_data {

			// Prepare the target vector (soft one-hot encoded, 6 outputs).
			targets := make([]float64, len(train_data))
			for i := range targets {
				targets[i] = 0.01
			}

			// Parse the label (0–5) and set the corresponding target to 0.99.
			label := int(train_data[i][0])
			targets[label] = 0.99

			/// Perform a training step using the input and target vectors
			// to update weights.
			net.Train(train_data[i][1:], targets)
		}

		// Log the completion of the current epoch.
		fmt.Printf("Finished epoch: %d", epochs)
	}

	// Calculate and print the total time taken for training.
	elapsed := time.Since(t1)
	fmt.Printf("Time taken to train: %s\n", elapsed)
	return nil
}
