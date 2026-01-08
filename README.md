🧠 Two-Layer Neural Network from Scratch (NumPy)

This repository contains a from-scratch implementation of a 2-layer neural network using only NumPy, without relying on high-level frameworks like TensorFlow or PyTorch.

The goal of this project is to understand the mathematics and mechanics of neural networks, including forward propagation, backpropagation, and gradient descent.

📌 Model Architecture
Input Layer (784 features)
        ↓
Hidden Layer (10 neurons, ReLU)
        ↓
Output Layer (10 neurons, Softmax)


Input: 28×28 grayscale images (flattened → 784 features)

Hidden layer: ReLU activation

Output layer: Softmax for multi-class classification (digits 0–9)

🧮 Mathematical Overview
Forward Propagation
𝑍
1
=
𝑊
1
𝑋
+
𝑏
1
Z
1
	​

=W
1
	​

X+b
1
	​

𝐴
1
=
ReLU
(
𝑍
1
)
A
1
	​

=ReLU(Z
1
	​

)
𝑍
2
=
𝑊
2
𝐴
1
+
𝑏
2
Z
2
	​

=W
2
	​

A
1
	​

+b
2
	​

𝐴
2
=
Softmax
(
𝑍
2
)
A
2
	​

=Softmax(Z
2
	​

)
Loss Function

Categorical Cross-Entropy (Log Loss)

𝐿
=
−
1
𝑚
∑
𝑦
log
⁡
(
𝑦
^
)
L=−
m
1
	​

∑ylog(
y
^
	​

)
Backpropagation

Gradients are computed using the chain rule:

𝑑
𝑍
2
=
𝐴
2
−
𝑌
one-hot
dZ
2
	​

=A
2
	​

−Y
one-hot
	​

𝑑
𝑊
2
=
1
𝑚
𝑑
𝑍
2
𝐴
1
𝑇
dW
2
	​

=
m
1
	​

dZ
2
	​

A
1
T
	​

𝑑
𝑏
2
=
1
𝑚
∑
𝑑
𝑍
2
db
2
	​

=
m
1
	​

∑dZ
2
	​

𝑑
𝑍
1
=
𝑊
2
𝑇
𝑑
𝑍
2
⊙
ReLU
′
(
𝑍
1
)
dZ
1
	​

=W
2
T
	​

dZ
2
	​

⊙ReLU
′
(Z
1
	​

)
𝑑
𝑊
1
=
1
𝑚
𝑑
𝑍
1
𝑋
𝑇
dW
1
	​

=
m
1
	​

dZ
1
	​

X
T
𝑑
𝑏
1
=
1
𝑚
∑
𝑑
𝑍
1
db
1
	​

=
m
1
	​

∑dZ
1
	​

🛠️ Implementation Details

Language: Python

Library: NumPy only

No ML frameworks used

Fully vectorized operations

Manual implementation of:

ReLU

Softmax

One-hot encoding

Backpropagation

Gradient descent

📂 Code Structure
.
├── init_params()        # Initialize weights & biases
├── forward_prop()       # Forward propagation
├── backward_prop()      # Backpropagation
├── update_params()      # Gradient descent update
├── one_hot()            # Label encoding
└── train loop

🚀 Training Loop (High Level)
for epoch in range(epochs):
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
    dW1, db1, dW2, db2 = backward_prop(...)
    W1, b1, W2, b2 = update_params(...)

🎯 Key Learning Outcomes

Why activation functions are required

Why matrix dimensions matter

How gradients flow backward

Why loss is averaged using 1/m

How a neural network actually learns

⚠️ Notes

This is an educational implementation, not optimized for speed

Initialization uses simple random values (no Xavier/He init)

Bias gradients are averaged across samples

📌 Why This Project?

Most beginners use deep learning libraries without understanding what happens underneath.
This project focuses on building intuition and mathematical clarity by implementing everything manually.

Note: A TensorFlow implementation of the same model is also included purely for comparison, to highlight how deep learning frameworks automate the same forward and backward propagation implemented manually in NumPy. 

 N-layer neural network has also been uploaded check my repos

👤 Author

Abraar
GitHub: Abraar77
