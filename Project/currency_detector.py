import numpy as np
import argparse
import cv2
import imutils

ap = argparse.ArgumentParser();
ap.add_argument("-i","--image",required=True)
ap.add_argument("--rupee",required=True)
args = vars(ap.parse_args())
print("Input rupee note : ",args["rupee"])

input_image = cv2.imread(args["image"])

aspect_ratio = 400.0/input_image.shape[1]
new_dimension = (400,int(input_image.shape[0]*aspect_ratio))

input_image = cv2.resize(input_image,new_dimension,cv2.INTER_LINEAR)
cv2.imshow("Resized Input Image",input_image)
cv2.waitKey(0)

gray_copy = cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray scale Image",gray_copy)
cv2.waitKey(0)

blurred_copy = cv2.GaussianBlur(gray_copy,(3,3),0)

cv2.imshow("Blurred",blurred_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
wide = cv2.Canny(blurred_copy,10,200)
cv2.imshow("Wide Edge Map",wide)
cv2.waitKey(0)

(_,cnts,_) = cv2.findContours(wide,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
contoured_copy = cv2.drawContours(input_image.copy(),cnts,-1,(0,0,255),4)
cv2.imshow("Contoured Image",contoured_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
max_area = 0
max_contour = cnts[0]
for (i,c) in enumerate(cnts):
    area = cv2.contourArea(c)
    if(area > max_area):
        area = max_area
        max_contour = c

#contoured_copy = cv2.drawContours(input_image,cnts,-1,(0,0,255),4)
(x,y,w,h) = cv2.boundingRect(max_contour)
clone = input_image.copy()
cv2.rectangle(clone,(x,y),(x+w, y+h),(0,0,255),4)
cv2.imshow("Bounded Image",clone)
cv2.waitKey(0)
cv2.destroyAllWindows()
new_input_image = input_image[y:(y+h),x:(x+w)]
cv2.imshow("Extracted Note", new_input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray_copy = cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)
