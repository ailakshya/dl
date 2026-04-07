import torch
import torch.nn as nn

X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[1],[1],[0]],         dtype=torch.float32)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

model = MLP()
opt = torch.optim.Adam(model.parameters(), lr=0.01)

for _ in range(2000):
    loss = nn.BCELoss()(model(X), y)
    opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    preds = (model(X) > 0.5).int().squeeze()
    accuracy = (preds == y.squeeze().int()).float().mean() * 100
    print("Predictions:", preds.tolist())
    print("Targets:    ", [0, 1, 1, 0])
    print(f"Accuracy:    {accuracy:.0f}%")
    print(f"Loss:        {loss:.4f}")
