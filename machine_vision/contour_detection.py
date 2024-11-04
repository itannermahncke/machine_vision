# pylint: disable=all
import cv2
import numpy

"""
Contour detection and drawing using different extraction modes to complement 
the understanding of hierarchies
"""
ERR_THRESH = 1

image = cv2.imread("image_1.jpg")
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
image_copy = image.copy()
for i, contour_primary in enumerate(contours):
    print(f"Contour {i}: {contour_primary}")
    segment_primary = (contour_primary[0][0], contour_primary[-1][0])
    for j, contour_sec in enumerate(contours):
        segment_secondary = (contour_sec[0][0], contour_sec[-1][0])
        # compare x
        for k in range(0,4):
            error = numpy.subtract(segment_primary[k % 2], segment_secondary[(k + 1) % 2])
            if error[0] < ERR_THRESH and error[1] < ERR_THRESH:

    
cv2.drawContours(image_copy, contours, 0, (0, 20*i, 0), 2, cv2.LINE_AA)
# see the results
cv2.imshow("EXTERNAL", image_copy)
print(f"EXTERNAL: {hierarchy}")
cv2.waitKey(0)
cv2.imwrite("contours_retr_external.jpg", image_copy)
cv2.destroyAllWindows()
