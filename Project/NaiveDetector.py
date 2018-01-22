import numpy as np
import argparse
import cv2
import imutils
import sys

print ("Simple Fake Note Detector\n----------------------------------")
print ("The Fake Note Detector will validate if the given rupee note is fake or real based on two conditions - ")
print ("1. The six digits present on the bottom right part of the note has to be increasing in height.")
print ("2. The six digits present on the top left part of the note has to be in increasing in height.")
cv2.waitKey(0)

ap = argparse.ArgumentParser();
ap.add_argument("-i","--image",required=True)
ap.add_argument("--rupee",required=True)
args = vars(ap.parse_args())

#Phase 1: Extracting the note from the image
print ("1. Inputing the Image")
input_image = cv2.imread(args["image"])

print ("2. Resizing the Image")
aspect_ratio = 500.0/input_image.shape[1]
new_dimension = (500,int(input_image.shape[0]*aspect_ratio))

input_image = cv2.resize(input_image,new_dimension,cv2.INTER_LINEAR)
cv2.imshow("Resized Input Image",input_image)
cv2.waitKey(0)

print ("3. Converting the image to Grayscale")
gray_copy = cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray scale Image",gray_copy)
cv2.waitKey(0)

print ("4. Blurring the Image")
blurred_copy = cv2.GaussianBlur(gray_copy,(3,3),0)

cv2.imshow("Blurred",blurred_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

print ("5. Detecting the edges")
wide = cv2.Canny(blurred_copy,10,200)
cv2.imshow("Wide Edge Map",wide)
cv2.waitKey(0)

print ("6. Finding the contours")
(_,cnts,_) = cv2.findContours(wide,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
contoured_copy = cv2.drawContours(input_image.copy(),cnts,-1,(0,0,255),1)
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

print ("7. Extracting the note")
(x,y,w,h) = cv2.boundingRect(max_contour)
clone = input_image.copy()
cv2.rectangle(clone,(x,y),(x+w, y+h),(0,0,255),2)
cv2.imshow("Bounded Image",clone)
cv2.waitKey(0)
cv2.destroyAllWindows()
new_input_image = input_image[y:(y+h),x:(x+w)]
if(h > w):
    print ("7.a Transposing the note")
    new_input_image = cv2.transpose(new_input_image)
    cv2.imshow("Transposed Note",new_input_image)
    cv2.waitKey(0)
    print ("7.b Flipping the note")
    new_input_image = cv2.flip(new_input_image,0)
cv2.imshow("Extracted Note", new_input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Phase 2: Extracting the regions in the note for validation
input_image = new_input_image
height = input_image.shape[0]
width = input_image.shape[1]

#1. Check the area at the bottom right
print ("8. Cropping out the bottom right part ")
right_bottom = input_image[height/2+70:height-15,width/2+30:width-50]

cv2.imshow("Right Bottom",right_bottom)
cv2.waitKey(0)

print ("9. Converting the image to Grayscale")
gray_copy = cv2.cvtColor(right_bottom,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray copy",gray_copy)
cv2.waitKey(0)

print ("10. Applying threshold to the image")
t, threshold_image = cv2.threshold(gray_copy, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
print ("11. Dilating the image")
threshold_image = cv2.dilate(threshold_image,None,1)

cv2.imshow("Thresholded image",threshold_image)
cv2.waitKey(0)
print ("12. Detecting the edges")
wide = cv2.Canny(threshold_image,10,200)
cv2.imshow("Wide",wide)
cv2.waitKey(0)
cv2.destroyAllWindows()
print ("13. Finding the contours")
digit_size_bottom = {}
(_,cnts,_) = cv2.findContours(wide,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
clone = right_bottom.copy()
print ("14. Extracting the height of the digit regions alone")
for (i,c) in enumerate(cnts):
    (x,y,w,h) = cv2.boundingRect(c)
    cv2.rectangle(clone,(x,y),(x+w,y+h),(0,0,255),1)
    if(x not in digit_size_bottom and h >= 12 and w > 10):
        digit_size_bottom[x] = h

aspect_ratio = 400.0/clone.shape[1]
new_dimension = (400,int(clone.shape[0]*aspect_ratio))

clone = cv2.resize(clone,new_dimension,cv2.INTER_LINEAR)
cv2.imshow("Contour",clone)
cv2.waitKey(0)
cv2.destroyAllWindows()
prev = -1
flag = True
count = 0
print ("15. Checking the heights to be in increasing order")
for key in sorted(digit_size_bottom.iterkeys()):
    if(count >= 6):
        break
    if(key >= 70 and prev == -1):
        prev = digit_size_bottom[key]
        count +=1
    elif(key >= 70):
        if(digit_size_bottom[key] < prev):
           flag = False
        count +=1
        prev = digit_size_bottom[key]

if(flag == False):
    print ("16. Displaying the result")
    cv2.putText(input_image,"Fake Note",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1.25,(255,0,0),4)
    cv2.imshow("Result",input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit()

#2. Check the area at top left
print ("16. Cropping out the top left part")
top_left = input_image[0+50:height/2-30,0:width/2-65]

cv2.imshow("Top Left",top_left)
cv2.waitKey(0)
print ("17. Converting the image to Grayscale")
gray_copy = cv2.cvtColor(top_left,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray copy",gray_copy)
cv2.waitKey(0)
print ("18. Blurring the image")
blurred_copy = cv2.GaussianBlur(gray_copy,(3,3),0)

print ("19. Detecting the edges")
wide = cv2.Canny(blurred_copy,80,200)
cv2.imshow("Wide",wide)
cv2.waitKey(0)
cv2.destroyAllWindows()

print ("20. Finding the contours")
digit_size_top = {}
(_,cnts,_) = cv2.findContours(wide,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
clone = top_left.copy()
print ("21. Extracting the height of digit regions alone")
for (i,c) in enumerate(cnts):
    (x,y,w,h) = cv2.boundingRect(c)
    cv2.rectangle(clone,(x,y),(x+w,y+h),(0,0,255),1)
    if(x not in digit_size_top and h >= 12 and w >= 8):
        digit_size_top[x] = h
aspect_ratio = 400.0/clone.shape[1]
new_dimension = (400,int(clone.shape[0]*aspect_ratio))

clone = cv2.resize(clone,new_dimension,cv2.INTER_LINEAR)

cv2.imshow("Contour",clone)
cv2.waitKey(0)
cv2.destroyAllWindows()
prev = -1
flag = True
count = 0
print ("22. Checking the heights to be in increasing order")
for key in sorted(digit_size_top.iterkeys()):
    if(count >= 6):
        break
    if(key >= 80 and prev == -1):
        prev = digit_size_top[key]
        count +=1
        #print (key , prev)
    elif(key >= 80):
        if(digit_size_top[key] < prev):
           flag = False
        prev = digit_size_top[key]
        count +=1
        #print (key, prev)

print ("23. Displaying the result")
if(flag == False):
    cv2.putText(input_image,"Fake Note",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1.25,(255,0,0),4)
    cv2.imshow("Result",input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit()
cv2.putText(input_image,"Real Note",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1.24,(255,0,0),4)
cv2.imshow("Result",input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
