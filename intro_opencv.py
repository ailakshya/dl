import cv2, numpy as np

img   = np.zeros((300,300,3), dtype=np.uint8)
cv2.circle(img, (150,150), 80, (255,255,255), -1)
edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)

assert edges.max() > 0
print("Test passed!")

cv2.imshow("Edges", edges); cv2.waitKey(0)
