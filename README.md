## Custom Neural Network visualized with websocket + <frontend-framework>
### Author: Bram Terlouw
### Start date: 15-04-2025
### Description:
This project strives to build en demonstrate a custom neural network. The neural network will be made from scratch using GoLang. To further demonstrate the neural network, a single-page frontend is developed to showcase the proces within the neural network. This is done real-time with the hulp of a websocket which sends updates to the frontend about active layers/nodes and values changed/transfered within the network while training/inference.

## Environment
- Podman 5.4.2
- Golang go1.24.2 windows/amd64
- <To-be-decided>

## Action Plan

### Go Backend Neural Network

#### Setup
- [x] Set up a containerized environment for the GoLang neural network.
- [x] Initialize a GoLang project within the containerized environment.
- [x] Implement a working WebSocket endpoint in the GoLang project.

#### Neural Network
- [x] Design a rough architecture of the neural network and its components for better understanding.
- [x] Build individual components/functions of the neural network with documentation:
  - [x] Utils (e.g., activation functions, matrix operations).
  - [x] Network (full neural network structure).
- [x] Construct a multilayer neural network.

#### Test Neural Network
- [x] Define a scenario to test the neural network.
- [x] Train the neural network.
- [x] Test the neural network with inference.

#### Visualize Neural Network
- [ ] Adjust code to log which layers and neurons are activated.
- [ ] Log values of all neurons after each activation.
- [ ] Add further visualization logging as needed (e.g., weights, biases).

#### Send Real-Time Updates
- [ ] Configure WebSocket to send updates on activated layers/neurons.
- [ ] Configure WebSocket to send updates on changed neuron values.
- [ ] Add additional WebSocket updates as needed (e.g., training progress).

### <frontend-framework> Single Page Application

#### Setup
- [ ] Set up a containerized environment for the <frontend-framework> application.
- [ ] Initialize a <frontend-framework> application within the containerized environment.

#### Visualization
- [ ] Create a rough design for the application and its components.
- [ ] Develop individual components for the application.
- [ ] Build a visualization of the neural network using these components.
- [ ] Connect the frontend to the WebSocket endpoint.
- [ ] Display activated layers and neurons in real-time.
- [ ] Show changed values of neurons in real-time.
- [ ] Add further visualizations as needed (e.g., network topology, loss metrics).