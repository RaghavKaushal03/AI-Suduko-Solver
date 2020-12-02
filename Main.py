
# line number 3-5 so that you wont get too many warnings in terminal
print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from functions import *
import sudukoSolver

# Changing Height and Width what we want so declaring Variables for that
# it should be SQUARE
pathImage = "Resources/1.jpg"
heightImg = 450
widthImg = 450
model = intializePredectionModel()  # Load the CNN Model


# Preparing the Image
img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))  # Resizing the image to make it square
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # creating a blank image for debugging purpose
# Note: cv2 has width,height and numpy has height,width.
imgThreshold = preProcess(img)

# Finding Contour and then biggest Contour
imgContours = img.copy()  # Image contour will have all the contours
imgBigContour = img.copy()  # Image Big Contour will have the biggest contour
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# RETR_External because we want to find external contour
cv2.drawContours(imgContours, contours, -1, (0, 0, 255), 3)  # Its BGR not RGB

# Biggest contour
biggest, maxArea = biggestContour(contours)  # FIND THE BIGGEST CONTOUR
print(biggest)
if biggest.size != 0:
    biggest = reorder(biggest)
    print(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25)  # Draw the biggest contour
    pts1 = np.float32(biggest)  # converting to float
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])   # this is how it should look
    matrix = cv2.getPerspectiveTransform(pts1, pts2) # GER
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

    # 4----> Splitting to find each digit i.e. 81 images
    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpColored)
    print(len(boxes))
    # cv2.imshow("Sample",boxes[65])
    numbers = getPredection(boxes, model)
    print(numbers)
    imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 0, 1)
    print(posArray)

    # 5------> Finding the Solution
    board = np.array_split(numbers,9)
    print(board)
    try:
        sudukoSolver.solve_sudoku(board)
    except:
        pass
    print(board)
    flatList = []
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    solvedNumbers = flatList*posArray
    imgSolvedDigits= displayNumbers(imgSolvedDigits,solvedNumbers)

    # 6------> Overlay that solution on original image
    pts2 = np.float32(biggest)  # preparing points for warp ie. changing them to float
    pts1 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])  # preparing points
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
    inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
    imgDetectedDigits = drawGrid(imgDetectedDigits)  # optional step (just to look good)
    imgSolvedDigits = drawGrid(imgSolvedDigits)

    # This is how we want our output to be .
    imageArray = ([img, imgThreshold, imgContours, imgBigContour],
                  [imgDetectedDigits, imgSolvedDigits, imgInvWarpColored, inv_perspective])
    # It joins all the image so that it becomes easy to understand
    stackedImage = stackImages(imageArray, 1)
    cv2.imshow('Stacked Images', stackedImage)

else:
    print("No Sudoku Found")
# so that output stays on screen until any key is pressed
cv2.waitKey(0)
