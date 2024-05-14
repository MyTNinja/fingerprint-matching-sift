import glob
import os
import cv2
import numpy as np

os.chdir("../database")
for file in glob.glob("*.bmp"):
    file_name = os.path.basename(file)
    img = cv2.imread(file, 0)

    bg = cv2.dilate(img, np.ones((5, 5), dtype=np.uint8))
    bg = cv2.GaussianBlur(bg, (5, 5), 1)
    src_no_bg = 255 - cv2.absdiff(img, bg)
    _, thresh = cv2.threshold(src_no_bg, 240, 255, cv2.THRESH_BINARY)

    thinned = cv2.ximgproc.thinning(thresh)

    cv2.imwrite('../template/{}_template.png'.format(file_name[:-4]), thinned)

cv2.waitKey(0)
cv2.destroyAllWindows()

