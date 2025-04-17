package neuralnetwork

import (
	"gonum.org/v1/gonum/mat"
)

type Layer struct {
	LayerName string     `json:"layer_name"`
	Neurons   []Neuron   `json:"neurons"`
	Weights   *mat.Dense `json:"weights,omitempty"`
}
