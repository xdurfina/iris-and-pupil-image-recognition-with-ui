import math, cv2, os
import numpy as np


############################
# Made by Jaroslav Ďurfina #
############################

# circle = [x,y,r]
# point = [x,y]

# Empty callback for createTrackbar fuction
def nothing(x):
    pass


# Function for creating GUI trackbars with initialized values
def setGUITrackbars(gaussianSize, gaussianSigma, canny1, canny2, accumulator, minDist, param1, param2, minRadius,
                    maxRadius, acceptance):
    cv2.createTrackbar("Gaussian Blur - Size", "IRIS AND PUPIL RECOGNITION", gaussianSize, 100, nothing)
    cv2.createTrackbar("Gaussian Blur - Sigma", "IRIS AND PUPIL RECOGNITION", gaussianSigma, 100, nothing)
    cv2.createTrackbar("Canny1", "IRIS AND PUPIL RECOGNITION", canny1, 100, nothing)
    cv2.createTrackbar("Canny2", "IRIS AND PUPIL RECOGNITION", canny2, 200, nothing)
    cv2.createTrackbar("Accumulator", "IRIS AND PUPIL RECOGNITION", accumulator, 50, nothing)
    cv2.createTrackbar("minDist", "IRIS AND PUPIL RECOGNITION", minDist, 50, nothing)
    cv2.createTrackbar("param1", "IRIS AND PUPIL RECOGNITION", param1, 150, nothing)
    cv2.createTrackbar("param2", "IRIS AND PUPIL RECOGNITION", param2, 100, nothing)
    cv2.createTrackbar("minRadius", "IRIS AND PUPIL RECOGNITION", minRadius, 100, nothing)
    cv2.createTrackbar("maxRadius", "IRIS AND PUPIL RECOGNITION", maxRadius, 300, nothing)
    cv2.createTrackbar("Acceptance", "IRIS AND PUPIL RECOGNITION", acceptance, 100, nothing)


# Function returns info about two circles ([x,y,r],[x,y,r])
# Return: state, message, d
def getCirclesIntersectionInfo(circle1, circle2):
    d = math.sqrt(math.pow(circle1[0] - circle2[0], 2) + math.pow(circle1[1] - circle2[1], 2))
    if d < (circle1[2] - circle2[2]):
        return 0, "Circle 2 is in circle 1", d
    elif d < (circle2[2] - circle1[2]):
        return 1, "Circle 1 is in circle 2", d
    elif d > (circle1[2] + circle2[2]):
        return 2, "Circles are the same", 0
    elif d < (circle1[2] + circle2[2]):
        return 3, "Circles intersect in two points", d
    else:
        return 4, "Circles do not intersect and are not in each other", -1


# Function returns area of union of two circles
def getAreaOfUnionOfTwoCircles(circle1, circle2):
    intersection = getAreaOfOverlapFromTwoCircles(circle1, circle2)
    areaCircle1 = math.pi * math.pow(circle1[2], 2)
    areaCircle2 = math.pi * math.pow(circle2[2], 2)
    return areaCircle1 + areaCircle2 - intersection


# Function returns area of overlap from two circles
def getAreaOfOverlapFromTwoCircles(circle1, circle2):
    if getCirclesIntersectionInfo(circle1, circle2)[0] == 0:
        return math.pi * math.pow(circle2[2], 2)

    if getCirclesIntersectionInfo(circle1, circle2)[0] == 1:
        return math.pi * math.pow(circle1[2], 2)

    if getCirclesIntersectionInfo(circle1, circle2)[0] == 2:
        return math.pi * math.pow(circle1[2], 2)

    if getCirclesIntersectionInfo(circle1, circle2)[0] == 3:
        info, message, d = getCirclesIntersectionInfo(circle1, circle2)
        d1 = (math.pow(circle1[2], 2) - math.pow(circle2[2], 2) + math.pow(d, 2)) / (2 * d)
        d2 = d - d1
        intersection = (math.pow(circle1[2], 2)) * math.acos(d1 / circle1[2]) - d1 * math.sqrt(
            math.pow(circle1[2], 2) - math.pow(d1, 2)) + (math.pow(circle2[2], 2)) * math.acos(
            d2 / circle2[2]) - d2 * math.sqrt(
            math.pow(circle2[2], 2) - math.pow(d2, 2))
        return intersection

    if getCirclesIntersectionInfo(circle1, circle2)[0] == 4:
        return 0


# Function return IoU from two circles
def getIoUfromTwoCircles(circle1, circle2):
    return getAreaOfOverlapFromTwoCircles(circle1, circle2) / getAreaOfUnionOfTwoCircles(circle1, circle2)


# Function returns if a given circle is zrenicka or duhovka based on IoU value
def isCircleDuhovkaOrZrenicka(circle, groundtruthduhovka, groundtruthzrenicka):
    if getIoUfromTwoCircles(circle, groundtruthduhovka) > getIoUfromTwoCircles(circle, groundtruthzrenicka):
        return 0  # Circle is duhovka
    else:
        return 1  # Circle is zrenicka


# Function returns masked image outside of given circle
def maskOutsideOfCircle(image, circle):
    h, w, ch = image.shape
    for y in range(0, h):
        for x in range(0, w):
            d = math.sqrt(math.pow(x - circle[0], 2) + math.pow(y - circle[1], 2))
            if d > circle[2]:
                image[y, x] = (0, 0, 0)

    return image


# Opening a named window and waiting for key response (1,2,3)
cv2.namedWindow('IRIS AND PUPIL RECOGNITION', cv2.WINDOW_AUTOSIZE)
key = cv2.waitKey(0)

# Setting GUI initial values based on selected image, also with ground truth values
# Function cv2.imread() with path opens the image and parameter 0 converts it to grayscale
if key == 49:
    setGUITrackbars(8, 2, 1, 1, 20, 4, 109, 88, 32, 120, 88)
    groundTruthZrenicka = [316, 162, 33]
    groundTruthDuhovka = [319, 163, 106]
    img = cv2.imread('eye1.jpg', 0)
elif key == 50:
    setGUITrackbars(1, 0, 0, 0, 15, 1, 108, 100, 0, 56, 10)
    groundTruthZrenicka = [155, 117, 24]
    groundTruthDuhovka = [155, 177, 56]
    img = cv2.imread('eye2.bmp', 0)
elif key == 51:
    setGUITrackbars(7, 2, 0, 0, 30, 1, 93, 95, 0, 215, 84)
    groundTruthZrenicka = [234, 230, 56]
    groundTruthDuhovka = [248, 232, 221]
    img = cv2.imread('eye3.jpg', 0)

# While true loop for infinite TrackbarPos value checking and image changing based on trackbar values
# Loop ends with pressing the ESC key
while True:
    # Function getTrackbarPos gets value from the given trackbar
    gcSize = cv2.getTrackbarPos("Gaussian Blur - Size", "IRIS AND PUPIL RECOGNITION")
    gcSigma = cv2.getTrackbarPos("Gaussian Blur - Sigma", "IRIS AND PUPIL RECOGNITION")
    threshold1 = cv2.getTrackbarPos("Canny1", "IRIS AND PUPIL RECOGNITION")
    threshold2 = cv2.getTrackbarPos("Canny2", "IRIS AND PUPIL RECOGNITION")
    accumulator = cv2.getTrackbarPos("Accumulator", "IRIS AND PUPIL RECOGNITION")
    minDist = cv2.getTrackbarPos("minDist", "IRIS AND PUPIL RECOGNITION")
    param1 = cv2.getTrackbarPos("param1", "IRIS AND PUPIL RECOGNITION")
    param2 = cv2.getTrackbarPos("param2", "IRIS AND PUPIL RECOGNITION")
    minRadius = cv2.getTrackbarPos("minRadius", "IRIS AND PUPIL RECOGNITION")
    maxRadius = cv2.getTrackbarPos("maxRadius", "IRIS AND PUPIL RECOGNITION")
    acceptance = cv2.getTrackbarPos("Acceptance", "IRIS AND PUPIL RECOGNITION")

    # Accumulator is very sensitive parameter that is used in HoughCircles function
    # Because of trackbar incapability to set float numbers, we divide the given number to get smaller one
    accumulator = accumulator / 10

    # Handling an error where parameter can not be zero
    if minDist == 0:
        minDist = 1

    if param1 == 0:
        param1 = 1

    if param2 == 0:
        param2 = 1

    # Handling an error where Size and Signa values can not be even
    if (gcSize % 2 == 0):
        gcSize = gcSize + 1

    # if (gcSigma % 2 == 0):
    #     gcSigma = gcSigma + 1

    # Image filtering functions
    imgAfterGaussianBlur = cv2.GaussianBlur(img, (gcSize, gcSize), gcSigma, borderType=cv2.BORDER_DEFAULT)
    ukazka = cv2.GaussianBlur(img, (gcSize, gcSize), gcSigma, borderType=cv2.BORDER_DEFAULT)

    imgAfterCanny = cv2.Canny(imgAfterGaussianBlur, threshold1, threshold2)

    imgAfterCvtColor = cv2.cvtColor(imgAfterGaussianBlur, cv2.COLOR_GRAY2BGR)

    # Function HoughCircles finds circles ([x,y,r]) on an image and returns the values to circles variable
    circles = cv2.HoughCircles(imgAfterGaussianBlur, cv2.HOUGH_GRADIENT, dp=accumulator, minDist=minDist,
                               param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    numberOfDuhovkas = 0
    numberOfZrenickas = 0
    arrayOfDuhovkas = []
    arrayOfZrenickas = []

    # Handling an error by checking if the circles variable is empty
    if circles is not None:
        # Changing the circles circle type so cv2.circle can work with it
        circles = np.uint16(np.around(circles))
        # Iterating through circles
        for i in circles[0, :]:
            # Checking if given circle is duhovka or zrenicka based on IoU value
            if (isCircleDuhovkaOrZrenicka(i, groundTruthDuhovka, groundTruthZrenicka)) == 0:  # Is Duhovka
                # Checking if circle meets the acceptance value
                if (getIoUfromTwoCircles(i, groundTruthDuhovka) > (acceptance / 100)):
                    numberOfDuhovkas += 1
                    arrayOfDuhovkas.append(i)
                    iouDuhovka = getIoUfromTwoCircles(i, groundTruthDuhovka)
                    iouDuhovka = round(iouDuhovka,2)
                    # Drawing the outer circle
                    cv2.circle(imgAfterCvtColor, (i[0], i[1]), i[2], (255, 0, 0), 2)
                    # Drawing the center of the circle
                    cv2.circle(imgAfterCvtColor, (i[0], i[1]), 2, (0, 0, 255), 3)
            elif (isCircleDuhovkaOrZrenicka(i, groundTruthDuhovka, groundTruthZrenicka)) == 1:  # Is Zrenicka
                if (getIoUfromTwoCircles(i, groundTruthZrenicka) > (acceptance / 100)):
                    numberOfZrenickas += 1
                    arrayOfZrenickas.append(i)
                    iouZrenicka = getIoUfromTwoCircles(i, groundTruthZrenicka)
                    iouZrenicka = round(iouZrenicka,2)
                    cv2.circle(imgAfterCvtColor, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    cv2.circle(imgAfterCvtColor, (i[0], i[1]), 2, (0, 0, 255), 3)

    # String declarations for cv2.putText function
    tpDuhovka = 'TP-DUHOVKA: '
    fpDuhovka = 'FP-DUHOVKA: '
    tpZrenicka = 'TP-ZRENICKA: '
    fpZrenicka = 'FP-ZRENICKA: '

    # Checking TP and FP values
    if numberOfDuhovkas > 0:
        tpDuhovka = tpDuhovka + "1"
        fpDuhovka = fpDuhovka + str(numberOfDuhovkas - 1)
    elif numberOfDuhovkas == 0:
        tpDuhovka = tpDuhovka + "0"
        fpDuhovka = fpDuhovka + "0"

    if numberOfZrenickas > 0:
        tpZrenicka = tpZrenicka + "1"
        fpZrenicka = fpZrenicka + str(numberOfZrenickas - 1)
    elif numberOfZrenickas == 0:
        tpZrenicka = tpZrenicka + "0"
        fpZrenicka = fpZrenicka + "0"

    if len(arrayOfDuhovkas) > 0 :
        iouDuhovka = "D - " + str(iouDuhovka)

    if len(arrayOfZrenickas) > 0:
        iouZrenicka = "Z - " + str(iouZrenicka)

    # Printing text on an image with TP and FP values
    imgAfterCvtColor = cv2.putText(imgAfterCvtColor, tpDuhovka, (00, 25), cv2.FONT_ITALIC, 1, (255, 0, 0),
                                   2, cv2.LINE_AA, False)
    imgAfterCvtColor = cv2.putText(imgAfterCvtColor, fpDuhovka, (00, 50), cv2.FONT_ITALIC, 1, (255, 0, 0),
                                   2, cv2.LINE_AA, False)
    imgAfterCvtColor = cv2.putText(imgAfterCvtColor, tpZrenicka, (00, 75), cv2.FONT_ITALIC, 1, (0, 255, 0),
                                   2, cv2.LINE_AA, False)
    imgAfterCvtColor = cv2.putText(imgAfterCvtColor, fpZrenicka, (00, 100), cv2.FONT_ITALIC, 1, (0, 255, 0),
                                   2, cv2.LINE_AA, False)

    if len(arrayOfDuhovkas) > 0:
        imgAfterCvtColor = cv2.putText(imgAfterCvtColor, iouDuhovka, (00, 125), cv2.FONT_ITALIC, 1, (255, 0, 0),
                                   2, cv2.LINE_AA, False)
    if len(arrayOfZrenickas) > 0:
        imgAfterCvtColor = cv2.putText(imgAfterCvtColor, iouZrenicka, (00, 150), cv2.FONT_ITALIC, 1, (0, 255, 0),
                                   2, cv2.LINE_AA, False)

    # ESC key condition for exiting the program
    k = cv2.waitKey(1)
    if k == 27:
        break

    # Image printing to an already created window
    cv2.imshow("IRIS AND PUPIL RECOGNITION.2", imgAfterCvtColor)
    cv2.imshow("IRIS AND PUPIL RECOGNITION.3", imgAfterCanny)
    #cv2.imshow("IRIS AND PUPIL RECOGNITION.5", ukazka)

    # BONUS: Applying the mask outside of duhovka
    # if numberOfDuhovkas == 1:
    #     cv2.imshow("IRIS AND PUPIL RECOGNITION.4", maskOutsideOfCircle(imgAfterCvtColor, arrayOfDuhovkas[0]))

