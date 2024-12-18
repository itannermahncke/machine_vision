# pylint: disable=all
import cv2
import numpy as np

"""
Contour detection and drawing using different extraction modes to complement 
the understanding of hierarchies
"""
SIZE_THRESH = 1000
MAX_THRESH = 100000

# get contours
image = cv2.imread("/home/ivymahncke/ros2_ws/src/machine_vision/images/clean_obj_1.JPG")
image = cv2.resize(image, (504 * 2, 378 * 2))
image = cv2.medianBlur(image, 5)

img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
image_copy = image.copy()

# create and draw bounding boxes
bounding_boxes = []  # tuple of contour, rect, box
for i, cnt in enumerate(contours):
    rect = cv2.minAreaRect(cnt)
    area = cv2.contourArea(cnt)
    # remove rects that are too small
    if SIZE_THRESH < area < MAX_THRESH:
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        bounding_boxes.append((cnt, rect, box))
        cv2.drawContours(image_copy, [box], 0, (0, 0, 255), 2, -1)

# see the results
cv2.imshow("EXTERNAL", image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
