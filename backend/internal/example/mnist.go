package example

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"time"

	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/nnet/data"
	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/nnet/network"
	"gonum.org/v1/gonum/mat"
)

func MnistTrain(net *network.Network) error {

	// Start timing the evaluation process.
	t1 := time.Now()

	// Open the MNIST training CSV file for reading.
	testFile, err := os.Open("file_path_to_train_mnist_dataset")
	if err != nil {
		return fmt.Errorf("failed to open mnist_train.csv: %w", err)
	}
	defer testFile.Close()

	// Train the network for a fixed number of epochs (x iterations over the
	// dataset).
	for epochs := 0; epochs < 5; epochs++ {

		// Initialize a CSV reader and skip the header row.
		reader := csv.NewReader(bufio.NewReader(testFile))
		_, err = reader.Read()
		if err != nil {
			return fmt.Errorf("failed to read header row: %w", err)
		}

		// Process each record (training example) in the CSV file.
		for {

			// Read a single row from the CSV file.
			record, err := reader.Read()
			if err == io.EOF {
				break // Exit the loop when the end of the file is reached.
			}

			// Print error when reader failed to read the training csv file.
			if err != nil {
				return fmt.Errorf("failed to read CSV record: %w", err)
			}

			// Prepare input for the forward pass.
			inputs, err := data.NormalizeInput(record, net.Layers[0])
			if err != nil {
				return fmt.Errorf("failed to normalize data: %w", err)
			}

			// Prepare the target vector (soft one-hot encoded, 10 outputs).
			targets := make([]float64, net.Layers[len(net.Layers)-1])
			for i := range targets {
				targets[i] = 0.01
			}

			// Parse the label (0–9) and set the corresponding target to 0.99.
			label, err := strconv.Atoi(record[0])
			if err != nil {
				return fmt.Errorf("failed to parse label: %w", err)
			}
			targets[label] = 0.99

			/// Perform a training step using the input and target vectors
			// to update weights.
			net.Train(inputs, targets)
		}

		// Log the completion of the current epoch.
		fmt.Printf("Finished epoch: %d", epochs)
	}

	// Calculate and print the total time taken for training.
	elapsed := time.Since(t1)
	fmt.Printf("Time taken to train: %s\n", elapsed)
	return nil
}

func MnistPredict(net *network.Network) error {

	// Start timing the evaluation process.
	t1 := time.Now()

	// Open the MNIST test CSV file for reading.
	checkFile, err := os.Open("path_to_test_file")
	if err != nil {
		return fmt.Errorf("failed to open mnist_test.csv: %w", err)
	}
	defer checkFile.Close()

	// Initialize a CSV reader and skip the header row.
	reader := csv.NewReader(bufio.NewReader(checkFile))
	// Skip the header row
	_, err = reader.Read()
	if err != nil {
		return fmt.Errorf("failed to read header row: %w", err)
	}

	// Initialize a counter for correct predictions.
	score := 0

	// Process each record (test example) in the CSV file.
	for {

		// Read a single row from the CSV file.
		record, err := reader.Read()
		if err == io.EOF {
			break // Exit the loop when the end of the file is reached.
		}

		// Print error when reader failed to read the training csv file.
		if err != nil {
			return fmt.Errorf("failed to read CSV record: %w", err)
		}

		// Prepare input for the forward pass.
		inputs, err := data.NormalizeInput(record, net.Layers[0])
		if err != nil {
			return fmt.Errorf("failed to normalize data: %w", err)
		}

		// Get the predicated value from the neural network.
		bestIdx, _ := predict(net, inputs)

		// Parse the true label (0–9) and compare it with the predicted class.
		target, err := strconv.Atoi(record[0])
		if err != nil {
			return fmt.Errorf("failed to parse label: %w", err)
		}

		// Increment the score if the prediction matches the true label.
		if bestIdx == target {
			score++
		}
	}

	// Calculate and print the total time taken for evaluation and the
	// number of correct predictions.
	elapsed := time.Since(t1)
	fmt.Printf("Time taken to check: %s\n", elapsed)
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
