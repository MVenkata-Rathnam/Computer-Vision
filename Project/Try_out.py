import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser();
ap.add_argument("-i","--image",required=True)
ap.add_argument("--rupee",required=True)
args = vars(ap.parse_args())
print("Input rupee note : ",args["rupee"])

input_image = cv2.imread(args["image"])
cv2.imshow("Input Image",input_image)
cv2.waitKey(0)

gray_copy = cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray scale Image",gray_copy)
cv2.waitKey(0)

count,binary_copy = cv2.threshold(gray_copy,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("Binary scale Image",binary_copy)

(_,cnts,_) = cv2.findContours(binary_copy,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
first_max_area = 0
second_max_area = 0
largest_contour = cnts[0]
second_largest_contour = cnts[0]
for (i,c) in enumerate(cnts):
    area = cv2.contourArea(c)
    if(area > first_max_area):
        second_max_area = first_max_area
        second_largest_contour = largest_contour
        first_max_area = area
        largest_contour = c
    elif(area > second_max_area):
        second_max_area = area
        second_largest_contour = c
print (first_max_area, second_max_area)
extracted_image = cv2.drawContours(input_image,second_largest_contour,-1,(0,255,0),5)
cv2.imshow("Contoured Image",extracted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print (len(cnts))


"""
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
"""

"""
box_dimension = cv2.minAreaRect(max_contour)
print(box_dimension,type(box_dimension))
box_dimension = np.int0(cv2.boxPoints(box_dimension))
cv2.drawContours(input_image,[box_dimension],-1,(0,0,255),2)
cv2.imshow("Input image",input_image)
print(box_dimension,type(box_dimension))
"""
