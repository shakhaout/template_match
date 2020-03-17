import os,glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import io
from skimage.measure import compare_ssim
os.getcwd()

img_list = []
for img in glob.glob('/........./*.JPG'):
    test_image = cv.imread(img)
    print(img)
    img_list.append(test_image)


template_data=[]
for myfile in glob.glob('/........./*.png'):
    image = cv.imread(myfile,0)
    print(myfile)
    template_data.append(image)
    
template_data=[x for x in template_data if x is not None]



for j in range(len(img_list)):
    # All the 6 methods for comparison in a list
    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    for meth in methods:
        img = img_list[j].copy()
        img_gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        method = eval(meth)
        # Apply template Matching
        for tmp in template_data:
            h,w = tmp.shape[:2]
            res = cv.matchTemplate(img_gray,tmp,method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv.rectangle(img,top_left, bottom_right, 255, 5)
        plt.figure(figsize=(20,12))
        plt.imshow(img,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()
        
        
        

