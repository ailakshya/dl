import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[1],[1],[0]],         dtype=torch.float32)

model = nn.Sequential(
    nn.Linear(2, 8), nn.ReLU(),
    nn.Linear(8, 1), nn.Sigmoid()
)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

for epoch in range(1000):
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    out = model(X)
    preds = (out > 0.5).int().squeeze()
    targets = y.squeeze().int()
    accuracy = (preds == targets).float().mean().item() * 100
    print("Predictions:", preds)
    print("Targets:    ", targets)
    print(f"Final Loss: {loss.item():.4f}")
    print(f"Accuracy:   {accuracy:.1f}%")
