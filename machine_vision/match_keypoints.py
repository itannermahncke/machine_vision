#!/usr/bin/env python3

""" A demo that shows how keypoint matches work using SIFT """

import cv2
import pickle
import numpy as np
from os import path


class KeyPointMatcherDemo(object):
    """KeyPointMatcherDemo shows the basics of interest point detection,
    descriptor extraction, and descriptor matching in OpenCV"""

    def __init__(self, im1_file, im2_file):
        enclosing_dir = path.dirname(path.realpath(__file__))
        print(enclosing_dir)
        self.im1_file = path.join(enclosing_dir, "../images", im1_file)
        self.im2_file = path.join(enclosing_dir, "../images", im2_file)

        self.keypoint_algorithm = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        self.im = None

        self.corner_threshold = 0.0
        self.ratio_threshold = 1.0

    def compute_matches(self):
        """reads in two image files and computes possible matches between them using SIFT"""
        im1 = cv2.imread(self.im1_file)
        im2 = cv2.imread(self.im2_file)
        im1 = cv2.resize(im1, (504 * 2, 378 * 2))
        im2 = cv2.resize(im2, (504 * 2, 378 * 2))

        im1_bw = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2_bw = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.keypoint_algorithm.detectAndCompute(im1_bw, None)
        kp2, des2 = self.keypoint_algorithm.detectAndCompute(im2_bw, None)

        matches = self.matcher.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            # make sure the distance to the closest match is sufficiently better than the second closest
            if (
                m.distance < self.ratio_threshold * n.distance
                and kp1[m.queryIdx].response > self.corner_threshold
                and kp2[m.trainIdx].response > self.corner_threshold
            ):
                good_matches.append((m.queryIdx, m.trainIdx))

        pts1 = np.zeros((len(good_matches), 2))
        pts2 = np.zeros((len(good_matches), 2))

        for idx in range(len(good_matches)):
            match = good_matches[idx]
            pts1[idx, :] = kp1[match[0]].pt
            pts2[idx, :] = kp2[match[1]].pt

        self.im = np.array(np.hstack((im1, im2)))

        # plot the points
        for i in range(pts1.shape[0]):
            cv2.circle(self.im, (int(pts1[i, 0]), int(pts1[i, 1])), 2, (255, 0, 0), 2)
            cv2.circle(
                self.im,
                (int(pts2[i, 0] + im1.shape[1]), int(pts2[i, 1])),
                2,
                (0, 0, 255),
                2,
            )
            cv2.line(
                self.im,
                (int(pts1[i, 0]), int(pts1[i, 1])),
                (int(pts2[i, 0] + im1.shape[1]), int(pts2[i, 1])),
                (0, 255, 0),
            )


def set_corner_threshold(thresh):
    """Sets the threshold to consider an interest point a corner.  The higher the value
    the more the point must look like a corner to be considered"""
    global matcher
    matcher.corner_threshold = thresh / 100000.0


def set_ratio_threshold(ratio):
    """Sets the ratio of the nearest to the second nearest neighbor to consider the match a good one"""
    global matcher
    matcher.ratio_threshold = ratio / 100.0


def mouse_event(event, x, y, flag, im):
    """Handles mouse events.  In this case when the user clicks, the matches are recomputed"""
    if event == cv2.EVENT_FLAG_LBUTTON:
        matcher.compute_matches()


if __name__ == "__main__":
    matcher = KeyPointMatcherDemo("clean_obj_1.JPG", "clean_obj_2.JPG")

    # setup a basic UI
    cv2.namedWindow("UI")
    cv2.createTrackbar("Corner Threshold", "UI", 0, 100, set_corner_threshold)
    cv2.createTrackbar("Ratio Threshold", "UI", 100, 100, set_ratio_threshold)
    matcher.compute_matches()

    cv2.imshow("MYWIN", matcher.im)
    cv2.setMouseCallback("MYWIN", mouse_event, matcher)

    while True:
        cv2.imshow("MYWIN", matcher.im)
        cv2.waitKey(50)
    cv2.destroyAllWindows()
