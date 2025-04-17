package neuralnetwork

type Neuron struct {
	Index  int `json:"index"`   // Index in layer
	Value  int `json:"value"`   // After activation
	PreAct int `json:"pre_act"` // Before activation
	Delta  int `json:"delta"`   // after backpropagation
}
