# Neural Network Training - Step-by-Step (with Matrix Placeholders)

A clean breakdown of how training works in a simple feedforward neural network containing using matrix operations and symbolic values. This network contains one input layer with 3 neurons, 1 hidden layer with 2 neurons and one output layer with one neuron. Ideal for learning, documentation, or implementation reference.

---

## Step 1: Convert input values into a matrix

Transform the input data vector into a column matrix:

```text
inputs = [
  [x1],
  [x2],
  [x3]
]
```

---

## Step 2: Calculate input for the hidden layer

Multiply the weights between the input and hidden layer with the input matrix:

```text
hiddenInputs = WeightsHidden × inputs

WeightsHidden = [
  [w11, w12, w13],
  [w21, w22, w23]
]

Result:
hiddenInputs = [
  [h1_input],
  [h2_input]
]
```

---

## Step 3: Calculate output of the hidden layer

Apply the sigmoid activation function element-wise:

```text
hiddenOutputs = sigmoid(hiddenInputs)

hiddenOutputs = [
  [h1_output],
  [h2_output]
]
```

---

## Step 4: Calculate input for the output layer

Multiply weights between hidden and output layer with the hidden outputs:

```text
finalInputs = WeightsOutput × hiddenOutputs

WeightsOutput = [
  [w31, w32]
]

Result:
finalInputs = [
  [o1_input]
]
```

---

## Step 5: Calculate final output of the network

Apply sigmoid again:

```text
finalOutputs = sigmoid(finalInputs)

finalOutputs = [
  [o1_output]
]
```

---

## Step 6: Convert target values into a matrix

```text
targets = [
  [t1]
]
```

---

## Step 7: Calculate error in the output layer

Subtract predicted output from the target value:

```text
outputErrors = targets - finalOutputs

outputErrors = [
  [e1]
]
```

---

## Step 8: Calculate error in the hidden layer

Distribute the output error back by multiplying with transposed output weights:

```text
hiddenErrors = WeightsOutputᵀ × outputErrors

WeightsOutputᵀ = [
  [w31],
  [w32]
]

hiddenErrors = [
  [e_h1],
  [e_h2]
]
```

---

## Step 9: Update weights (hidden to output)

1. Compute gradient:
```text
outputGradient = outputErrors ⊙ sigmoidPrime(finalOutputs)
```

2. Get weight change:
```text
deltaWeightsOutput = outputGradient × hiddenOutputsᵀ
```

3. Apply learning rate and update:
```text
WeightsOutput += learningRate × deltaWeightsOutput
```

---

## Step 10: Update weights (input to hidden)

1. Compute gradient:
```text
hiddenGradient = hiddenErrors ⊙ sigmoidPrime(hiddenOutputs)
```

2. Get weight change:
```text
deltaWeightsHidden = hiddenGradient × inputsᵀ
```

3. Apply learning rate and update:
```text
WeightsHidden += learningRate × deltaWeightsHidden
```

---

## 🪨 Legend
- `⊙` = element-wise multiplication  
- `Aᵀ` = transpose of matrix A  
- `sigmoidPrime(x)` = derivative of sigmoid function  
- `+=` = add the result to the existing weights

---

This README provides a symbolic, implementation-agnostic view of what happens behind the scenes during one training step of a simple feedforward neural network.

