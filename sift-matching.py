import numpy as np
import cv2
import glob
import os

MIN_MATCH_COUNT = 15

input_img = cv2.imread('./input/100__M_Left_ring_finger_CR.bmp', 0)
sift = cv2.SIFT_create()

bg = cv2.dilate(input_img, np.ones((5, 5), dtype=np.uint8))
bg = cv2.GaussianBlur(bg, (5, 5), 1)
src_no_bg = 255 - cv2.absdiff(input_img, bg)
_, thresh = cv2.threshold(src_no_bg, 240, 255, cv2.THRESH_BINARY)

img = cv2.ximgproc.thinning(thresh)
kp = sift.detect(img, None)
img1 = cv2.drawKeypoints(img, kp, img)

flag = 0

os.chdir("./template")
for file in glob.glob("*.png"):

    frame = cv2.imread(file, 0)
    sift = cv2.SIFT_create()
    kp = sift.detect(frame, None)
    img2 = cv2.drawKeypoints(frame, kp, frame)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        print("Matched " + str(file))

        flag = 1

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
        cv2.imshow("Match", img3)
        cv2.imwrite('../output/matched_{}.png'.format(str(file)[:-13]), img3)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        matchesMask = None

if flag == 0:
    print("No Matches among the given set!!")
