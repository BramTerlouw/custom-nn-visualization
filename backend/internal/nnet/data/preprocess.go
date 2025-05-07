package data

import (
	"fmt"
	"math"
)

// Preprocess_CSV
// Read Csv data file, normalize if necessary, and
// create OneHotEncoded targets.
//
// Arguments:
//   - file_path:				file location to download.
//   - skip_header_first_row:	bool to skip first record.
//   - normalize:				bool to normalize.
//   - targetIdx:				index of target col.
//   - outputSize:				output size of network.
//
// Returns:
//   - data:					float64 matrix with data.
//   - targets:					float64 array with targets.
//   - err:						(optional).
func Preprocess_CSV(file_path string, skip_header_first_row, normalize bool, targetIdx, outputSize int) ([][]float64, [][]float64, error) {

	// Read the csv file.
	data, err := ReadCSV(file_path, skip_header_first_row, targetIdx)
	if err != nil {
		return nil, nil, err
	}

	// Normalize if necessary.
	if normalize {
		data, err = matrix_normalize(data)
		if err != nil {
			return nil, nil, err
		}
	}

	// Generate targat matrix.
	targets, err := oneHotEncode(data, outputSize)
	if err != nil {
		return nil, nil, err
	}

	return data, targets, nil
}

func Preprocess_Float64_Matrix(data [][]float64, do_normalize bool, outputSize int) ([][]float64, [][]float64, error) {

	// Define output without any preprocessing.
	output := data

	// Normalize if necessary.
	if do_normalize {
		normalized_data, err := matrix_normalize(data)
		if err != nil {
			return nil, nil, err
		}
		output = normalized_data
	}

	// Generate targat matrix.
	targets, err := oneHotEncode(data, outputSize)
	if err != nil {
		return nil, nil, err
	}

	return output, targets, nil
}

func matrix_normalize(data [][]float64) ([][]float64, error) {

	// Loop over every item in the matrix, skipping the label.
	for rowIdx, row := range data {

		for colIdx := 1; colIdx < len(row); colIdx++ {

			// Normalize and assign
			entry := row[colIdx]
			data[rowIdx][colIdx] = (entry / 255.0 * 0.99) + 0.01
		}
	}
	return data, nil
}

func oneHotEncode(data [][]float64, outputSize int) ([][]float64, error) {

	var target_matrix [][]float64

	// Loop over every row (training entry).
	for rowIdx, row := range data {

		targets := make([]float64, outputSize)
		for i := range targets {

			// Assign all entries value 0.01.
			targets[i] = 0.01
		}

		// Check if the target (label) is whole number.
		if math.Mod(row[0], 1.0) != 0 {
			return nil, fmt.Errorf("target in training row is not a whole number on row %d", rowIdx)
		}

		// Convert label to its index and set value to 0.99.
		label := int(row[0])
		targets[label] = 0.99
		target_matrix = append(target_matrix, targets)
	}

	return target_matrix, nil
}
