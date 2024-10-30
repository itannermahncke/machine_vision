# pylint: disable=all
import cv2

"""
Contour detection and drawing using different extraction modes to complement 
the understanding of hierarchies
"""

image2 = cv2.imread("image_1.jpg")
image2 = cv2.convertScaleAbs(image2, alpha=5.0, beta=0)
img_gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
ret, thresh2 = cv2.threshold(img_gray2, 150, 255, cv2.THRESH_BINARY)

contours4, hierarchy4 = cv2.findContours(
    thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
image_copy5 = image2.copy()
for i, contour in enumerate(contours4):
    print(f"{i} weiner goofy haha silly cum devil")
    cv2.drawContours(image_copy5, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA)

# see the results
cv2.imshow("EXTERNAL", image_copy5)
print(f"EXTERNAL: {hierarchy4}")
cv2.waitKey(0)
cv2.imwrite("contours_retr_external.jpg", image_copy5)
cv2.destroyAllWindows()
