import torch
import torch.nn as nn
import torch.optim as optim

# --- Data ---
# XOR truth table: output is 1 only when the two inputs differ
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)  # 4 input pairs
y = torch.tensor([[0],[1],[1],[0]],         dtype=torch.float32)  # corresponding XOR labels

# --- Model ---
# XOR is not linearly separable, so we need a hidden layer to learn it.
# Architecture: Input(2) -> Hidden(8, ReLU) -> Output(1, Sigmoid)
model = nn.Sequential(
    nn.Linear(2, 8), nn.ReLU(),   # hidden layer: expands to 8 neurons, ReLU adds non-linearity
    nn.Linear(8, 1), nn.Sigmoid() # output layer: collapses to 1 value in [0, 1]
)

# Adam optimizer with a learning rate of 0.01
optimizer = optim.Adam(model.parameters(), lr=0.01)
# Binary Cross-Entropy loss, suitable for binary classification
criterion = nn.BCELoss()

# --- Training loop ---
for epoch in range(1000):
    pred = model(X)               # forward pass: compute predictions
    loss = criterion(pred, y)     # compute loss between predictions and true labels
    optimizer.zero_grad()         # clear gradients from the previous step
    loss.backward()               # backpropagate: compute gradients
    optimizer.step()              # update weights using the computed gradients

# --- Evaluation ---
with torch.no_grad():             # disable gradient tracking (not needed for inference)
    out = model(X)
    print("Predictions:", (out > 0.5).int().squeeze())  # threshold at 0.5 to get binary output
    print("Targets:    ", y.squeeze().int())
    print(f"Final Loss: {loss.item():.4f}")
