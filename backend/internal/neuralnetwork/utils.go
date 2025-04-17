package neuralnetwork

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// Sigmoid activation function
// Performs a calculation on the inputs of the neuron (weight x input).
// Returns a new value as input for the next layer.
func Sigmoid(r, c int, z float64) float64 {
	return 1.0 / (1 + math.Exp(-1*z))
}

// Sigmoid derivative function
// Calculates the gradient of the sigmoid activation for backpropagation.
// Uses the output of the sigmoid function to compute: sigmoid(x) * (1 - sigmoid(x)).
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
	return Element_multiply(m, Subtract(ones_matrix, m))
}

// Matrix multiplication
// Multiplies matrix m by matrix n.
// The number of columns in m must equal the number of rows in n.
// Returns a new matrix with dimensions (rows of m) × (columns of n).
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
// Returns a new matrix with dimensions (rows of m) x (cols of m)
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
// Returns a new matrix with values modified by the input function.
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
// Returns a new matrix with multiplied values.
func Scale(s float64, m mat.Matrix) mat.Matrix {

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
// Returns a new matrix with dimensions (rows of m) x (cols of m)
func Add(m, n mat.Matrix) mat.Matrix {

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
// Returns a new matrix with dimensions (rows of m) x (cols of m)
func Subtract(m, n mat.Matrix) mat.Matrix {

	// Get rows and cols of first matrix, generate new matrix.
	rows, cols := m.Dims()
	output := mat.NewDense(rows, cols, nil)

	// Perfom subtraction with all corresponding elements in the matrices.
	output.Sub(m, n)
	return output
}
