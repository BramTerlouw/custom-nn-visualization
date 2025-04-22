package logger

type ActivationLog struct {
	Epoch            int               `json:"epoch"`
	InputIndex       int               `json:"input_index"`
	InputValues      []float64         `json:"input_values"`
	LayerActivations []LayerActivation `json:"layer_activations"`
}

type LayerActivation struct {
	Layer             int                `json:"layer"`
	NeuronActivations []NeuronActivation `json:"neuron_activations"`
	WeightsBefore     [][]float64        `json:"weights_before"`
	WeightsAfter      [][]float64        `json:"weights_after"`
}

type NeuronActivation struct {
	Neuron     int     `json:"neuron"`
	Activation float64 `json:"activation"`
}
