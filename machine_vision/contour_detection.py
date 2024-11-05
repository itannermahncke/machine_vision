# pylint: disable=all
import cv2
import numpy as np

"""
Contour detection and drawing using different extraction modes to complement 
the understanding of hierarchies
"""

image = cv2.imread("image_1.jpg")
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
)
image_copy = image.copy()
for i, cnt in enumerate(contours):
    rect = cv2.minAreaRect(cnt)
    area = cv2.contourArea(cnt)
    # remove rects that are too small
    if area > 1000:
        print(f"Rect: {rect}\n   Width: {rect[1][0]} Height: {rect[1][1]}\n   Area: {area}")
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(image_copy,[box],0,(0,0,255),2, -1)

    
# cv2.drawContours(image_copy, contours, 0, (0, 20*i, 0), 2, cv2.LINE_AA)
# see the results
cv2.imshow("EXTERNAL", image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()