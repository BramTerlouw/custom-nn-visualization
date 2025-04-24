package main

import (
	"fmt"

	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/example"
	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/nnet/model"
	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/nnet/network"
)

func main() {

	execute_train_model()
	// execute_tiny_rand_example()
	// execute_minst_example()

	// // Default HTTP route
	// http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
	// 	_, err := w.Write([]byte("Neural Network Backend"))
	// 	if err != nil {
	// 		log.Printf("Error writing response: %v", err)
	// 	}
	// })
	// // WebSocket route
	// http.HandleFunc("/ws", websocket.HandleConnection)

	// log.Println("Starting server on :8080")
	// if err := http.ListenAndServe(":8080", nil); err != nil {
	// 	log.Fatalf("Server failed: %v", err)
	// }
}

func execute_train_model() {

	// ARGS/ENVS
	INPUT_SIZE := 784
	HIDDEN_LAYER_SIZES := []int{100}
	OUTPUT_SIZE := 10

	layers := append([]int{INPUT_SIZE}, HIDDEN_LAYER_SIZES...)
	layers = append(layers, OUTPUT_SIZE)

	// Create a network for MNIST: 784 inputs, 100 hidden, 10 outputs
	net := network.NewNetwork(layers, 0.01)

	err := model.TrainFeedForwardModel(net, "C:/Users/bramt/Downloads/mnist_train.csv", "C:/Users/bramt/Downloads/mnist_test.csv", 5)
	if err != nil {
		fmt.Print(err)
	}
}

func execute_minst_example() {

	// ARGS/ENVS
	INPUT_SIZE := 784
	HIDDEN_LAYER_SIZES := []int{12, 12, 12, 10}
	OUTPUT_SIZE := 10

	layers := append([]int{INPUT_SIZE}, HIDDEN_LAYER_SIZES...)
	layers = append(layers, OUTPUT_SIZE)

	// Create a network for MNIST: 784 inputs, 100 hidden, 10 outputs
	net := network.NewNetwork(layers, 0.01)

	// Train the network
	if err := example.MnistTrain(net); err != nil {
		fmt.Printf("Training failed: %v\n", err)
		return
	}

	// Test the network
	if err := example.MnistPredict(net); err != nil {
		fmt.Printf("Prediction failed: %v\n", err)
		return
	}
}

func execute_tiny_rand_example() {

	// ARGS/ENVS
	INPUT_SIZE := 5
	HIDDEN_LAYER_SIZES := []int{10, 20, 10}
	OUTPUT_SIZE := 6

	layers := append([]int{INPUT_SIZE}, HIDDEN_LAYER_SIZES...)
	layers = append(layers, OUTPUT_SIZE)

	// Create a network for MNIST: 784 inputs, 100 hidden, 10 outputs
	net := network.NewNetwork(layers, 0.01)

	// Train the network
	if err := example.TinyRandTrain(net); err != nil {
		fmt.Printf("Training failed: %v\n", err)
		return
	}
}
