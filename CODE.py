import cv2
import numpy as np
import cv2 as cv

img1 = None
img2 = None
contour = None

img1 =  cv.imread("img1.jpg",1)
img2 =  cv.imread("img2.jpg",1)
cv.imshow("Image 1",img1)
cv.imshow("Image 2",img2)
cv.waitKey(0)
cv.destroyAllWindows()

img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Fixed: Removed invalid keyword arguments
blurred1 = cv2.GaussianBlur(gray1, (5,5), 0)
blurred2 = cv2.GaussianBlur(gray2, (5,5), 0)

diff = cv.absdiff(blurred1, blurred2)
_, thresh_diff = cv.threshold(diff, 50, 255, cv.THRESH_BINARY)
contours, _  = cv.findContours(thresh_diff, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for contour in contours:
    if cv.contourArea(contour) > 500:
        (x, y, w, h) = cv.boundingRect(contour)
        cv.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv.imshow("Highlighted Differences", img1)
cv.waitKey(0)
cv.destroyAllWindows()
