import cv2

img = cv2.imread("image.png")
edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
assert edges.max() > 0
cv2.imshow("Edges", edges); cv2.waitKey(0)
