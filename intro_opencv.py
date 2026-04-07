import cv2, torch, torch.nn as nn
img = cv2.imread("image.png")
g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
t = torch.tensor(cv2.resize(g,(8,8))/255., dtype=torch.float32).flatten().unsqueeze(0)
print(nn.Linear(64,1)(t).item()); cv2.imshow("",cv2.Canny(g,100,200)); cv2.waitKey(0)
