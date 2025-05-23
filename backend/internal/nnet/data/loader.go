package data

import (
	"encoding/csv"
	"os"
	"strconv"
)

// ReadCSV
// Read Csv data file into memory, skip label header (if
// necessary) and set target col as first col.
//
// Arguments:
//   - file_path:				file location to download.
//   - skip_header_first_row:	bool to skip first record.
//   - targetIdx:				index of target col.
//
// Returns:
//   - data:					float64 matrix with data.
//   - err:						(optional).
func ReadCSV(file_path string, skip_header_first_row bool, targetIdx int) ([][]float64, error) {

	// Check if file can be opened.
	file, err := os.Open(file_path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Read the whole csv into memory.
	reader := csv.NewReader(file)
	raw, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	// Set starting index (skip one if header exists).
	startIdx := 0
	if skip_header_first_row {
		startIdx = 1
	}

	// Loop through the records.
	var data [][]float64
	for _, data_row := range raw[startIdx:] {

		var row []float64

		// Extract the label value at targetIdx
		labelVal, err := strconv.ParseFloat(data_row[targetIdx], 64)
		if err != nil {
			return nil, err
		}

		// Put the label at the beginning
		row = append(row, labelVal)

		// Loop through all entrys in row
		for idx, data_entry := range data_row {

			// Skip the target column.
			if idx == targetIdx {
				continue
			}

			// Parse to float.
			val, err := strconv.ParseFloat(data_entry, 64)
			if err != nil {
				return nil, err
			}
			row = append(row, val)
		}
		data = append(data, row)
	}

	return data, nil
}
