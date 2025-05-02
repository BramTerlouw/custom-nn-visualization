package main

import "github.com/BramTerlouw/custom-nn-visualization/backend/internal/example"

func main() {

	example.TrainMnist()
	// example.TrainRandMatrix()

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
