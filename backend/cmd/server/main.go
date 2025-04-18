package main

import (
	"log"
	"net/http"

	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/neuralnetwork"
	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/websocket"
)

func main() {

	// Create and test NN
	nn := neuralnetwork.NewNetwork([]int{2, 3, 6, 12, 6, 3, 2}, 1.2)
	inputs := []float64{1.0, 3.0}
	nn.Forward(inputs)

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
