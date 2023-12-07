import cv2
import pytesseract
import numpy as np
from PIL import Image
import math
import os

output_path = r'output.txt'
input_path = r'input.png'

def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    px = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    py = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if d == 0:
        return None
    return np.array([px / d, py / d])

def detectParticipantList( path ):
    img = cv2.imread(path)
    cv2.imshow('img', img)
    cv2.waitKey(0)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect edges using the Canny algorithm
    edges = cv2.Canny(gray, 500, 1000, apertureSize=3)
    

    #Detect lines using the Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=300, maxLineGap=50)

    # Draw the detected lines on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        #color purple
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
    
    cv2.imshow('img', img)
    cv2.waitKey(0)


    # Select the longest vertical line (assumed to be the edge of the participant list)
    v1 = None
    v2 = None
    max_len = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < 10:
            if abs(y1 - y2) > max_len:
                v1 = np.array([x1, y1])
                v2 = np.array([x2, y2])
                max_len = abs(y1 - y2)

    #display the longest vertical line
    cv2.line(img, tuple(v1), tuple(v2), (0, 0, 255), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)

    # Compute the intersection points of the two lines with the top and bottom edges of the image


    tl = line_intersection(v1.tolist() + [0, 0], [0, 0, 0, img.shape[0]])
    bl = line_intersection(v1.tolist() + [0, img.shape[0]], [0, 0, 0, img.shape[0]])

    # Crop the image to the rectangular area defined by the two corners and return it as the participant list
    participant_list = img[int(tl[1]):int(bl[1]), int(v1[0]):]
    return participant_list

def detectCameraOnIcons( participant_list, iconPath):
    icon = cv2.imread(iconPath, cv2.IMREAD_COLOR)

    # Detect the camera icon using template matching with colors
    participant_list.astype(np.float32)
    res = cv2.matchTemplate(participant_list, icon, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)

    # Draw a rectangle around the detected camera icons
    for point in zip(*loc[::-1]):
        cv2.rectangle(participant_list, point, (point[0] + icon.shape[1], point[1] + icon.shape[0]), (0, 0, 255), 2)
    
    return participant_list, loc

def getParticipantsWhoseCameraOn(participant_list, loc, iconPath):
    icon = cv2.imread(iconPath, cv2.IMREAD_COLOR)
    names = []
    for point in zip(*loc[::-1]):
        roi = participant_list[point[1]-5:point[1]+ icon.shape[0] + 5, icon.shape[1]*2:point[0]-icon.shape[1]+5]
        name = pytesseract.image_to_string(roi, lang='eng', config='--psm 7')
        names.append(name)
        print(name)
    
    #write the names to a file
    with open(output_path, 'w') as f:
        for name in names:
            f.write("%s\n" % name)

    

# Load the screenshot
path = '' #path to the screenshot
dpi120icon = '' #path to the 120dpi camera icon
dpi96icon = '' #path to the 96dpi camera icon

img = Image.open(path)
if math.ceil(img.info['dpi'][0]) == 120:
    iconPath = dpi120icon
    print("120")
else:
    iconPath = dpi96icon
    print("96")



participant_list = detectParticipantList(path)
participant_list_with_icons, icon_locations = detectCameraOnIcons(participant_list, iconPath)
getParticipantsWhoseCameraOn(participant_list, icon_locations, iconPath)

cv2.destroyAllWindows()