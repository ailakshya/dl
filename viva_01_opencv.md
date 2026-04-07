# Viva Preparation — Experiment 01: Introduction to Deep Learning and OpenCV

---

## 1. What is OpenCV?
**OpenCV** (Open Source Computer Vision Library) is a library used for image and video processing.
- Written in C++, works with Python
- Used for: reading images, filtering, edge detection, object detection, etc.

---

## 2. What is Deep Learning?
Deep Learning is a subset of Machine Learning where **neural networks with many layers** learn patterns from data automatically.
- Input → Hidden Layers → Output
- Used for: image recognition, speech, NLP, etc.

---

## 3. What does our code do? (Step by Step)

### Step 1 — Create a test image
```python
img = np.zeros((300,300,3), dtype=np.uint8)
cv2.circle(img, (150,150), 80, (255,255,255), -1)
```
- `np.zeros` → creates a black canvas of size 300x300 with 3 color channels (BGR)
- `cv2.circle` → draws a white filled circle at center (150,150) with radius 80

### Step 2 — Convert to Grayscale
```python
cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```
- Converts 3-channel color image (BGR) → 1-channel grayscale
- **Why?** Edge detection works on single-channel images

### Step 3 — Edge Detection (Canny)
```python
cv2.Canny(gray, 100, 200)
```
- Finds edges by detecting sharp changes in pixel intensity
- `100` = lower threshold, `200` = upper threshold
- Pixels above 200 → strong edge, below 100 → not an edge, between → edge only if connected to a strong edge

### Step 4 — Test
```python
assert edges.max() > 0
```
- Checks that at least one edge was detected
- If no edges found, code crashes with AssertionError

---

## 4. Key Concepts

### What is an image in OpenCV?
- An image is a **NumPy array** of pixel values
- Color image shape: `(height, width, 3)` — 3 channels: Blue, Green, Red
- Grayscale shape: `(height, width)` — 1 channel, values 0–255

### Why BGR and not RGB?
- OpenCV uses **BGR** (Blue, Green, Red) by default — historical reason from its C++ origins

### What is Canny Edge Detection?
A 4-step process:
1. **Gaussian Blur** — smooths image to remove noise
2. **Gradient calculation** — finds intensity changes
3. **Non-maximum suppression** — thins the edges
4. **Hysteresis thresholding** — keeps strong edges, discards weak ones

### What is a threshold?
A cutoff value. Pixels above the threshold are kept, below are discarded.

---

## 5. Possible Viva Questions & Answers

**Q: What is the shape of a color image in OpenCV?**
A: `(height, width, 3)` — 3 channels in BGR order.

**Q: Why do we convert to grayscale before edge detection?**
A: Canny works on single-channel images. Grayscale reduces the 3 channels to 1, simplifying intensity comparison.

**Q: What do the two numbers in Canny (100, 200) mean?**
A: Lower and upper thresholds. Values above 200 are strong edges, below 100 are discarded, between 100–200 are kept only if connected to a strong edge.

**Q: What does `np.zeros` do?**
A: Creates an array filled with zeros — a completely black image.

**Q: What is `dtype=np.uint8`?**
A: Unsigned 8-bit integer — pixel values range from 0 (black) to 255 (white).

**Q: What is the difference between deep learning and OpenCV?**
A: OpenCV uses hand-crafted algorithms (like Canny). Deep learning **learns** the features automatically from data using neural networks.

**Q: What is the role of OpenCV in deep learning?**
A: OpenCV is used for **preprocessing** — reading, resizing, converting images before feeding them into a deep learning model.

**Q: What does `assert` do?**
A: Tests a condition. If it's False, it raises an `AssertionError` and stops the program — used for quick testing.

**Q: What is `cv2.waitKey(0)`?**
A: Waits indefinitely until a key is pressed before closing the image window.

---

## 6. Key Terms to Remember

| Term | Meaning |
|---|---|
| Pixel | Smallest unit of an image |
| Grayscale | Single-channel image (0=black, 255=white) |
| BGR | Blue-Green-Red color format used by OpenCV |
| Edge | Boundary between two regions of different intensity |
| Threshold | Cutoff value to decide edge or not |
| NumPy array | How images are stored in Python |
| Canny | Algorithm to detect edges in an image |

---

## 7. One-line Summary
> OpenCV reads and processes images as NumPy arrays. We created a test image, converted it to grayscale, and applied Canny edge detection to find the boundaries of shapes.
