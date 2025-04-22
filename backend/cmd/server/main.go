package main

import (
	"fmt"
	"log"
	"net/http"

	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/example"
	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/neuralnetwork"
	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/websocket"
)

func main() {

	// Create a network for MNIST: 784 inputs, 100 hidden, 10 outputs
	net := neuralnetwork.NewNetwork([]int{784, 100, 10}, 0.01)

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

	// Default HTTP route
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		_, err := w.Write([]byte("Neural Network Backend"))
		if err != nil {
			log.Printf("Error writing response: %v", err)
		}
	})
	// WebSocket route
	http.HandleFunc("/ws", websocket.HandleConnection)

	log.Println("Starting server on :8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
