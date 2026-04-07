import cv2, numpy as np

# Create test image: white rectangle on black canvas
img = np.zeros((300,300,3), dtype=np.uint8)
cv2.rectangle(img, (50,50), (250,250), (0,255,0), -1)  # green rectangle

gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur    = cv2.GaussianBlur(gray, (5,5), 0)             # reduce noise
edges   = cv2.Canny(blur, 50, 150)

assert edges.max() > 0
print("Test passed!")

cv2.imshow("Original", img)
cv2.imshow("Blur", blur)
cv2.imshow("Edges", edges)
cv2.waitKey(0)
