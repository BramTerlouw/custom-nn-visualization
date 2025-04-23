package network

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

// Sigmoid activation function
// Performs a calculation on the inputs of the neuron (weight x input).
//
// Arguments:
//   - r: rows, unused
//   - c: cols, unused
//   - z: float value to perform activation function on.
//
// Returns:
//   - Returns a new value as input for the next layer.
func Sigmoid(r, c int, z float64) float64 {
	return 1.0 / (1 + math.Exp(-1*z))
}

// Sigmoid derivative function
// Calculates the gradient of the sigmoid activation for backpropagation.
// Uses the output of the sigmoid function to compute: sigmoid(x) * (1 - sigmoid(x)).
//
// Arguments:
//   - m: Matrix used to calculate the derivetive of sigmoid.
//
// Returns:
//   - Returns a new matrix with the derivetive values of the input matrix values.
func SigmoidPrime(m mat.Matrix) mat.Matrix {

	// Get the rows from input and create float64 arraw with size of the rows.
	rows, _ := m.Dims()
	ones_array := make([]float64, rows)

	// Fill the float64 array with values '1' and create new matrix from it.
	for i := range ones_array {
		ones_array[i] = 1
	}
	ones_matrix := mat.NewDense(rows, 1, ones_array)

	// Perform sigmoid(x) * (1 - sigmoid(x)) where input m = hiddenoutputs = sigmoid(x)
	return Element_multiply(m, Subtract_matrix(ones_matrix, m))
}

// Matrix multiplication
// Multiplies matrix m by matrix n.
// The number of columns in m must equal the number of rows in n.
// Multiplication takes place like m[r, i] x n[i, c]
//
// Arguments:
//   - m: Matrix used to perform elementwise multiplication on.
//   - n: matrix used to perform elementwise multiplication with.
//
// Returns:
//   - Returns a new matrix with dimensions (rows of m) × (columns of n).
func Matrix_multiply(m, n mat.Matrix) mat.Matrix {

	// Get rows and cols of first matrix, generate new matrix.
	rows, _ := m.Dims()
	_, cols := n.Dims()
	output := mat.NewDense(rows, cols, nil)

	// Perform matrix multiplication.
	output.Product(m, n)
	return output
}

// Element multiplication
// Multiplies each element in m by the corresponding element in n.
// The rows and columns must be equal in both matrices.
//
// Arguments:
//   - m: Matrix used to perform elementwise multiplication on.
//   - n: matrix used to perform elementwise multiplication with.
//
// Returns:
//   - Returns a new matrix with dimensions (rows of m) x (cols of m)
func Element_multiply(m, n mat.Matrix) mat.Matrix {

	// Get rows and cols of first matrix, generate new matrix.
	rows, cols := m.Dims()
	output := mat.NewDense(rows, cols, nil)

	// Perform elementwise multiplication.
	output.MulElem(m, n)
	return output
}

// Apply function on elements
// Performs the input function on all elements in the input matrix.
//
// Arguments:
//   - fn: function to be applied.
//   - m: matrix of which all values are to be used in the function fn.
//
// Returns:
//   - Returns a new matrix with values modified by the input function.
func Apply_fn(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {

	// Get rows and cols of first matrix, generate new matrix.
	rows, cols := m.Dims()
	output := mat.NewDense(rows, cols, nil)

	// Perform the function on all elements.
	output.Apply(fn, m)
	return output
}

// Multiply with float
// Perform a multiplication with float s on all elements in input matrix.
//
// Arguments:
//   - s: float value used for multiplication.
//   - m: matrix of which all values are to be multiplied with s.
//
// Returns:
//   - Returns a new matrix with multiplied values.
func Scale_matrix(s float64, m mat.Matrix) mat.Matrix {

	// Get rows and cols of first matrix, generate new matrix.
	rows, cols := m.Dims()
	output := mat.NewDense(rows, cols, nil)

	// Perform a multiplication with float s on all elements.
	output.Scale(s, m)
	return output
}

// Element addition
// Add each element in m with the corresponding element in n.
// The rows and columns must be equal in both matrices.
//
// Arguments:
//   - m: matrix to which addition takes place.
//   - n: matrix used for the addition.
//
// Returns:
//   - Returns a new matrix with dimensions (rows of m) x (cols of m)
func Add_matrix(m, n mat.Matrix) mat.Matrix {

	// Get rows and cols of first matrix, generate new matrix.
	rows, cols := m.Dims()
	output := mat.NewDense(rows, cols, nil)

	// Perfom addition with all corresponding elements in the matrices.
	output.Add(m, n)
	return output
}

// Element subtraction
// Add each element in m by the corresponding element in n.
// The rows and columns must be equal in both matrices.
//
// Arguments:
//   - m: matrix from which subtraction takes place.
//   - n: matrix used for the subtraction.
//
// Returns:
//   - Returns a new matrix with dimensions (rows of m) x (cols of m)
func Subtract_matrix(m, n mat.Matrix) mat.Matrix {

	// Get rows and cols of first matrix, generate new matrix.
	rows, cols := m.Dims()
	output := mat.NewDense(rows, cols, nil)

	// Perfom subtraction with all corresponding elements in the matrices.
	output.Sub(m, n)
	return output
}

// Creates random float64 values.
// Values are uniformly distributed between -1/sqrt(v) and 1/sqrt(v),
// which helps maintain stable variance during weight initialization.
//
// Arguments:
//   - v: used to determine the scaling factor for the distribution.
//
// Returns:
//   - float64 containing a uniformly distributed value.
func randomFloat(v float64) (data float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}
	return dist.Rand()
}
