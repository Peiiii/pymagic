import cv2
import numpy as np

img = cv2.imread('data/2.jpg')
result1 = img.copy()
result2 = img.copy()
result3 = img.copy()

img = cv2.GaussianBlur(img,(3,3),0)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray,50,150,apertureSize = 3)
cv2.imwrite("canny.jpg", edges)
#hough transform
lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength=90,maxLineGap=10)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(result1,(x1,y1),(x2,y2),(0,0,255),1)
    print (x1,y1)
    print (x2,y2)
cv2.circle(result2,(207,151),5,(0,255,0),5)
cv2.circle(result2,(517,285),5,(0,255,0),5)
cv2.circle(result2,(17,601),5,(0,255,0),5)
cv2.circle(result2,(343,731),5,(0,255,0),5)

cv2.imwrite("result1.jpg", result1)
cv2.imwrite("result2.jpg", result2)

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
m = cv2.getPerspectiveTransform(src, dst)
result = cv2.warpPerspective(result3, m, (337, 488))
cv2.imwrite("result.jpg", result)
cv2.imshow("result", result)
cv2.waitKey(0)
