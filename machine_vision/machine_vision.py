def main():
    print('Hi from machine_vision.')
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


if __name__ == '__main__':
    main()
