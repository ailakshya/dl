# Viva — Experiment 01: Intro to Deep Learning & OpenCV

---

## What is OpenCV?
Open Source Computer Vision Library — processes images/videos as NumPy arrays. Used for filtering, edge detection, preprocessing for deep learning.

## What is Deep Learning?
Neural networks with many layers that **learn features automatically** from data (unlike OpenCV which uses hand-crafted algorithms).

```
Raw Image → OpenCV (preprocess) → Deep Learning Model → Prediction
```

---

## Code Walkthrough

```python
import cv2

img   = cv2.imread("image.png")                                        # load image → NumPy array (H,W,3) BGR
edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)    # grayscale → edge detection
print("Pass!" if edges.max() > 0 else "Fail!")                         # test: at least one edge found
cv2.imshow("Edges", edges); cv2.waitKey(0)                             # show result, wait for keypress
```

---

## Functions

| Function | What it does |
|---|---|
| `cv2.imread(file)` | Reads image → NumPy array `(H,W,3)`, BGR order. Returns `None` if not found |
| `cv2.cvtColor(img, code)` | Converts color space. `COLOR_BGR2GRAY` → single channel |
| `cv2.Canny(img, t1, t2)` | Edge detection. `t1`=low threshold, `t2`=high threshold |
| `cv2.imshow(name, img)` | Opens window to display image |
| `cv2.waitKey(0)` | Waits for keypress (0 = forever). Keeps window open |

---

## Canny — 4 Steps
1. **Gaussian Blur** — removes noise
2. **Sobel Gradient** — finds intensity changes (X & Y)
3. **Non-max Suppression** — thins edges to 1px
4. **Hysteresis Threshold** — above t2 = edge, below t1 = discard, between = edge if connected to strong edge

---

## Image Shapes

| Type | Shape | Values |
|---|---|---|
| Color | `(H, W, 3)` | 0–255 per channel |
| Grayscale | `(H, W)` | 0–255 |
| Canny output | `(H, W)` | 0 or 255 |

**Why BGR?** OpenCV's historical default (not RGB). Convert with `COLOR_BGR2RGB` for matplotlib.

---

## Viva Q&A

**Q: Why grayscale before Canny?**
Canny needs single-channel. Grayscale removes color, keeps brightness (where edges are).

**Q: What do Canny thresholds (100, 200) mean?**
Above 200 = strong edge. Below 100 = not an edge. Between = edge only if connected to a strong edge. Rule: upper = 2× lower.

**Q: What is `cv2.waitKey(0)`?**
Waits indefinitely for a keypress — required to keep the imshow window open.

**Q: OpenCV vs Deep Learning?**
OpenCV = hand-crafted rules. Deep learning = learns features from data automatically.

**Q: What does `assert edges.max() > 0` check?**
That at least one edge pixel (255) was found. If all zeros, no edges detected — test fails.

---

## Key Line Explained

```python
edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
```

Two functions chained — inner runs first:

**Step 1 — `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`**
- Converts color image `(H,W,3)` → grayscale `(H,W)`
- Formula: `gray = 0.114B + 0.587G + 0.299R`

**Step 2 — `cv2.Canny(..., 100, 200)`**
- Takes grayscale output, runs 4 steps: blur → gradient → thin → threshold
- `100` = lower threshold, `200` = upper threshold
- Output: 255 at edges, 0 elsewhere

---

## Where is Deep Learning in this code?

**It's not.** This experiment is OpenCV only — hand-crafted algorithms, no learning.

| This code | Deep Learning |
|---|---|
| Rules written by humans | Learns rules from data |
| Canny = fixed math | CNN = learned filters |
| Same output always | Improves with more data |

Experiment 01 builds the image processing foundation. From Experiment 03 (CNNs) onwards, the model learns features automatically instead of using Canny manually.

---

## One-liner
> OpenCV loads images as NumPy arrays; Canny finds edges by detecting sharp intensity changes using a 4-step algorithm with double thresholding.
