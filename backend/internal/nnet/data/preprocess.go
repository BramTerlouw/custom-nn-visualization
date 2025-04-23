package data

import (
	"fmt"
	"strconv"
)

// Normalize the input
// Normalizes an array of string values for more efficient
// usage in the neural network.
//
// Arguments:
//   - record: string array to normalize.
//   - inputSize: number of values the nn expects as input.
//
// Returns:
//   - Returns a tuple with an array with normalized float64 values
//     and error|nil.
func NormalizeInput(record []string, inputSize int) ([]float64, error) {

	// Check the size of the input record against the expected input size
	if len(record) < inputSize {
		return nil, fmt.Errorf("length of record did not match inputsize: expected %d, got %d", inputSize, len(record))
	}

	// Normalizing pixel values from [0, 255] to [0.01, 0.99].
	inputs := make([]float64, inputSize)
	for i := 1; i < len(record); i++ {

		// Parse the pixel value as a float64.
		record_value, err := strconv.ParseFloat(record[i], 64)
		if err != nil {
			return nil, fmt.Errorf("failed to parse input %d: %w", i, err)
		}

		// Normalize the pixel value to the range [0.01, 0.99].
		inputs[i-1] = (record_value / 255.0 * 0.99) + 0.01
	}

	return inputs, nil
}
