# Viva — Experiment 01: Intro to Deep Learning & OpenCV

---

## What is OpenCV?
Open Source Computer Vision Library — processes images/videos as NumPy arrays. Used for filtering, edge detection, and preprocessing for deep learning.

## What is Deep Learning?
Neural networks with learned weights that **automatically learn features** from data. Unlike OpenCV which uses fixed hand-crafted algorithms.

---

## Full Code

```python
import cv2, torch, torch.nn as nn
img = cv2.imread("image.png")
g   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
t   = torch.tensor(cv2.resize(g,(8,8))/255.).flatten().unsqueeze(0)
print(nn.Linear(64,1)(t).item()); cv2.imshow("",cv2.Canny(g,100,200)); cv2.waitKey(0)
```

---

## Every Component Explained

### `import cv2, torch, torch.nn as nn`
- `cv2` — OpenCV for image processing
- `torch` — PyTorch for tensors and deep learning
- `torch.nn` — neural network building blocks (layers, activations)

---

### `cv2.imread("image.png")`
- Reads image from disk → NumPy array of shape `(H, W, 3)`
- Color order is **BGR** (not RGB)
- Returns `None` if file not found

---

### `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`
- Converts 3-channel BGR image → 1-channel grayscale
- Formula: `gray = 0.114B + 0.587G + 0.299R`
- Output shape: `(H, W)` — single channel
- **Why?** Both Canny and the DL layer need a simpler single-channel input

---

### `cv2.resize(g, (8,8))`
- Shrinks the grayscale image to 8×8 pixels = 64 values
- **Why 8×8?** The `nn.Linear(64,1)` layer needs a fixed input size of exactly 64

---

### `/ 255.`
- Normalizes pixel values from range `0–255` → `0.0–1.0`
- **Why?** Neural networks train and perform better with small normalized values

---

### `torch.tensor(...)`
- Converts NumPy array → PyTorch tensor
- Required to pass data into any PyTorch model or layer

---

### `.flatten()`
- Reshapes 2D array `(8,8)` → 1D array of 64 values
- `nn.Linear` expects a flat 1D input, not a 2D grid

---

### `.unsqueeze(0)`
- Adds a batch dimension: `(64,)` → `(1, 64)`
- PyTorch layers always expect input shape `(batch_size, features)`
- Here batch size = 1 (single image)

---

### `nn.Linear(64, 1)`
- A single fully-connected neural network layer
- Takes 64 inputs → produces 1 output
- Has **weights** and **bias** (randomly initialized here)
- This is the deep learning component — a real neuron
- Formula: `output = weight × input + bias`

---

### `.item()`
- Extracts the scalar value from a PyTorch tensor
- Converts `tensor([0.43])` → `0.43` (plain Python float)

---

### `cv2.Canny(g, 100, 200)`
- Detects edges using the Canny algorithm
- `100` = lower threshold, `200` = upper threshold
- 4 steps: Gaussian blur → gradient → non-max suppression → hysteresis threshold
- Output: binary image — 255 at edges, 0 elsewhere

---

### `cv2.imshow("", edges)`
- Opens a window displaying the edge image
- `""` = empty window title

---

### `cv2.waitKey(0)`
- Waits indefinitely for a keypress before closing the window

---

## Why is DL not very useful here?

`nn.Linear` here is **untrained** — weights are random, so the output number is meaningless. Real DL needs training data and a loss function to learn something useful.

| Component | Type | Useful? |
|---|---|---|
| `cv2.Canny` | OpenCV | Yes — detects real edges |
| `nn.Linear` | Deep Learning | No — random weights, no training |

**Experiment 01 is about learning the tools.** DL becomes meaningful from Experiment 02 (MLP) onwards.

---

## Viva Q&A

**Q: What does `cv2.imread` return?**
A: A NumPy array of shape `(H, W, 3)` in BGR format.

**Q: Why convert to grayscale?**
A: Canny and the linear layer need single-channel input. Grayscale reduces 3 channels to 1.

**Q: Why resize to 8×8?**
A: `nn.Linear(64,1)` needs exactly 64 inputs. 8×8 = 64 pixels.

**Q: Why divide by 255?**
A: Normalizes pixel values to 0–1. Neural networks work better with small values.

**Q: What does `.unsqueeze(0)` do?**
A: Adds batch dimension. PyTorch expects input shape `(batch, features)`.

**Q: What is `nn.Linear`?**
A: A fully connected layer. Computes `output = weight × input + bias`.

**Q: What are the Canny thresholds?**
A: 100 = lower, 200 = upper. Above 200 = strong edge, below 100 = no edge, between = edge only if connected to a strong edge.

**Q: Is this real deep learning?**
A: Technically yes — `nn.Linear` is a neural network layer. But it's untrained so the output is meaningless. Real DL requires training.

---

## One-liner
> OpenCV loads and processes the image; Canny detects edges; a single untrained `nn.Linear` layer demonstrates the basic DL component — real DL begins from Experiment 02.
