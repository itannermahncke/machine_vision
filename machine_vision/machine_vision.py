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

        # Converts images to grayscale
        im1_bw = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2_bw = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        # Detects and computes the keypoints for eahc image
        kp1, des1 = self.keypoint_algorithm.detectAndCompute(im1_bw, None)
        kp2, des2 = self.keypoint_algorithm.detectAndCompute(im2_bw, None)

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

        # creates new points using the good matches
        pts1 = np.zeros((len(good_matches), 2))
        pts2 = np.zeros((len(good_matches), 2))
        for idx in range(len(good_matches)):
            match = good_matches[idx]
            pts1[idx, :] = kp1[match[0]].pt
            pts2[idx, :] = kp2[match[1]].pt

        # creates new image to contain all the points
        self.im = np.array(np.hstack((im1, im2)))

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


if __name__ == "__main__":
    main()
