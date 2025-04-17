package main

import (
	"github.com/BramTerlouw/custom-nn-visualization/backend/internal/neuralnetwork"
)

func main() {

	nn := neuralnetwork.NewNetwork(2, 3, 2, 0.1)

	inputs := []float64{1.0, 3.0}
	targets := []float64{1.1, 3.2}
	nn.Train(inputs, targets)
	// _ = nn.Forward(inputs)

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
