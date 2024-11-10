import cv2
import pickle
import numpy as np
from os import path


class KeypointMatch(object):
    """
    KeypointMatch class
    """

    def __init__(self, im1_filepath, im2_filepath):
        """
        Initializes a new keypointMatch object
        """
        self.keypoint_algorithm = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        self.im = None
        self.size_thresh = 100
        self.size_tresh_max = 100000

        self.corner_threshold = 0.0
        self.ratio_threshold = 1.0

        self.img1 = im1_filepath
        self.img2 = im2_filepath

    def compute_matches(self):
        """
        tbd
        """

        # Gets the images
        im1 = cv2.imread(self.img1)
        im2 = cv2.imread(self.img2)

        im1 = cv2.resize(im1, (378 * 2, 504 * 2))
        im2 = cv2.resize(im2, (378 * 2, 504 * 2))

        # Converts images to grayscale
        im1_bw = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2_bw = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        # Detects and computes the keypoints for each image
        kp1, des1 = self.keypoint_algorithm.detectAndCompute(im1_bw, None)
        kp2, des2 = self.keypoint_algorithm.detectAndCompute(im2_bw, None)

        # Find contours in each image
        im1_bw = cv2.medianBlur(im1_bw, 5)
        im2_bw = cv2.medianBlur(im2_bw, 5)
        ret1, thresh1 = cv2.threshold(im1_bw, 20, 255, cv2.THRESH_BINARY)
        # ret2, thresh2 = cv2.threshold(im2_bw, 20, 255, cv2.THRESH_BINARY)
        contours1, hierarchy1 = cv2.findContours(
            thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        # contours2, hierarchy2 = cv2.findContours(
        #     thresh2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        # )
        im1_cpy = im1.copy()
        im2_cpy = im2.copy()

        # create bounding boxes for each group of contours
        b_box1 = []
        # b_box2 = []
        filtered_contours = []
        for i, cnt in enumerate(contours1):
            rect1 = cv2.minAreaRect(cnt)
            area1 = cv2.contourArea(cnt)
            # remove rects that are too small
            if area1 > self.size_thresh and area1 < self.size_tresh_max:
                filtered_contours.append(cnt)
                box = cv2.boxPoints(rect1)
                box = np.intp(box)
                b_box1.append((cnt, rect1, box))
                cv2.drawContours(im1_cpy, [box], 0, (0, 0, 255), 2, -1)
        # for i, cnt in enumerate(contours2):
        #     rect2 = cv2.minAreaRect(cnt)
        #     area2 = cv2.contourArea(cnt)
        #     # remove rects that are too small
        #     if area2 > self.size_thresh:
        #         box = cv2.boxPoints(rect2)
        #         box = np.intp(box)
        #         b_box2.append((cnt, rect2, box))
        #         cv2.drawContours(im2_cpy, [box], 0, (0, 0, 255), 2, -1)

        # filters out keypoints that dont fall within a bounding box

        # for i, kp in enumerate(kp2):
        #     for cnt in contours2:
        #         if cv2.pointPolygonTest(cnt, (kp.pt[0], kp.pt[1]), False):
        #             kp_bbox2.append(kp)

        # Finds the matches of keypoints across images
        matches = self.matcher.knnMatch(des1, des2, k=2)

        # Sorts through the matches to filter out 'good matches' that are better for use
        good_matches = []
        for m, n in matches:
            # make sure the distance to the closest match is sufficiently better than the second closest
            if (
                m.distance < self.ratio_threshold * n.distance
                and kp1[m.queryIdx].response > self.corner_threshold
                and kp2[m.trainIdx].response > self.corner_threshold
            ):
                good_matches.append((m.queryIdx, m.trainIdx))

        matches_1 = []
        matches_2 = []
        for pair in good_matches:
            matches_1.append(pair[0])
            matches_2.append(pair[1])

        kp_inx = [-1] * len(good_matches)
        print(good_matches)
        # kp_bbox2 = []
        for j, cnt in enumerate(filtered_contours):
            for i, mat in enumerate(good_matches):
                if i in matches_1:
                    if (
                        cv2.pointPolygonTest(
                            cnt,
                            (kp1[matches_1[i]].pt[0], kp1[matches_1[i]].pt[1]),
                            False,
                        )
                        >= 0
                    ):
                        kp_inx[i] = j
                        print(f"i:{i} j:{j}")
                    else:
                        print("failed ppt")

        print(kp_inx)

        # creates new points using the good matches
        pts1 = np.zeros((len(good_matches), 2))
        pts2 = np.zeros((len(good_matches), 2))
        for idx in range(len(good_matches)):
            if kp_inx[idx] != -1:
                match = good_matches[idx]
                pts1[idx, :] = kp1[match[0]].pt
                pts2[idx, :] = kp2[match[1]].pt

        # creates new image to contain all the points
        self.im = np.array(np.hstack((im1_cpy, im2)))

        # plots the points
        for i in range(pts1.shape[0]):
            cv2.circle(self.im, (int(pts1[i, 0]), int(pts1[i, 1])), 2, (255, 0, 0), 2)
            cv2.circle(
                self.im,
                (int(pts2[i, 0] + im1.shape[1]), int(pts2[i, 1])),
                2,
                (255, 0, 0),
                2,
            )
            cv2.line(
                self.im,
                (int(pts1[i, 0]), int(pts1[i, 1])),
                (int(pts2[i, 0] + im1.shape[1]), int(pts2[i, 1])),
                (0, 255, 0),
            )


def main():
    print("Hi from machine_vision.")
    # get an image
    # draw contours around each object in the image
    # for each contour, see if endpoint matches with another contour
    # if so, merge them into one contour
    # this should result in sealed boxes
    # use dimensions of contour to group internal keypoints

    # get a second image
    # note a given translation between the two images (eg +50px X)
    # repeat all the prior steps for the second image
    # apply the given translation to each original keypoint (predicted keypoints)
    # for each predicted keypoint, find the closest keypoint in the new image
    # label it with the same contour/object name

    # display both images with superimposed and labeled contours/keypoint groups

    matching = KeypointMatch("images/simple_1.jpg", "images/simple_2.jpg")
    matching.compute_matches()
    cv2.imshow("EXTERNAL", matching.im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
