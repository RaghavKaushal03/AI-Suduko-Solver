import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Tip: You can see all the details of predefined function by ctrl + click on them

# Model which will recognize digits from 1 to 9
def intializePredectionModel():
    model = load_model('Resources/myModel.h5')
    return model

# 1.Preprocessing the Image
def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert Image to Gray Image
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # Convert Gray Image to Blur Image
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)  # Convert Blur image into Threshold Image
    # RAGHAV ------> Can add image dilation and image erosion <---------------
    return imgThreshold


# 2.So that all the points are in order (0,0), (width,0), (0,height), (width,height)
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)  # This will sum all the values
    myPointsNew[0] = myPoints[np.argmin(add)]  # The min value will be 1st point eg-:(0,0)
    myPointsNew[3] =myPoints[np.argmax(add)]  # The max value will be 4th point eg-:(50,50)
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]  # One value will be negative
    myPointsNew[2] = myPoints[np.argmax(diff)]  # One value will be positive
    return myPointsNew


# 3.Finding the biggest contour
def biggestContour(contours):
    biggest = np.array([])  # An array which will have all the corner points
    max_area = 0
    for i in contours:  # checks each contour one by one
        area = cv2.contourArea(i)  # Finds area of selected contour
        if area > 50:
            peri = cv2.arcLength(i, True)  # True is whether its closed or not
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)  # this find no. of points(vertices)
            if area > max_area and len(approx) == 4:  # len of approx is 4 means rectangle or square
                biggest = approx  # biggest will contain all the corner points
                max_area = area
    return biggest, max_area


# 4.To split full image into 81 images
def splitBoxes(img):
    rows = np.vsplit(img, 9)    # Vertical split into 9 parts
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)   # each of above 9 parts is horizontally split into 9 parts again.
        for box in cols:
            boxes.append(box)  # add 81 images into boxes array
    return boxes


# 5.To get the predictions of each number one by one
def getPredection(boxes, model):
    result = []
    for image in boxes:  # so we will individually perform task 81 times
        # Prepare image
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] -4]
        img = cv2.resize(img, (28, 28))
        img = img / 255
        img = img.reshape(1, 28, 28, 1)
        # Getting the predictions
        predictions = model.predict(img)
        classIndex = model.predict_classes(img)
        probabilityValue = np.amax(predictions)
        # Save to Result
        if probabilityValue > 0.8:  # This means we are able to predict that its not empty space
            result.append(classIndex[0])
        else:
            result.append(0)  # else its zero
    return result


# 6.To display images as well as prepare images for empty blocks
def displayNumbers(img, numbers, color=(0, 255, 0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range(0, 9):
        for y in range(0, 9):
            if numbers[(y*9)+x] != 0:
                 cv2.putText(img, str(numbers[(y*9)+x]),
                               (x*secW+int(secW/2)-10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img


#### 6 - To draw grids between warp perspective (OPTIONAL)
def drawGrid(img):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for i in range (0,9):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1],secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i,img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0),2)  # just drawing lines
        cv2.line(img, pt3, pt4, (255, 255, 0),2)
    return img


# 6.To stack all the images in one window
def stackImages(imgArray, scale):
    # here scale is whether you have to increase the size or decrease the size
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    return ver
