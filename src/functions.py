import cv2
import numpy as np
from classes import *
from copy import deepcopy
from PIL import Image
from collections import Counter

# Opens the file ------------------------------------------------------------------
def openFile(path):
    img = Image.open(path)
    img.show()

# Finding the best match for a character ------------------------------------------
def match(img, templates, startP, endP, threshold):
    locationCount = -1
    optimizedLocation = []
    scale = 1
    x = []
    y = []
    for scale in [i/100.0 for i in range(startP, endP + 1, 3)]:
        locations = []
        countLocation = 0

        for template in templates:
            if (scale*template.shape[0] > img.shape[0] or scale*template.shape[1] > img.shape[1]):
                continue

            template = cv2.resize(template, None,
                fx = scale, fy = scale, interpolation = cv2.INTER_CUBIC)
            result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            result = np.where(result >= threshold)
            countLocation += len(result[0])
            locations += [result]

        x.append(countLocation)
        y.append(scale)
        if (countLocation > locationCount):
            locationCount = countLocation
            optimizedLocation = locations
            scale = scale
        elif (countLocation < locationCount):
            pass
    return optimizedLocation, scale

# Finding the width of spacing and lines used for staffs --------------------------
def lengthReference(img):
    rowNumber = img.shape[0]
    columnNumber = img.shape[1]

    whiteRunList = []
    blackRunList = []
    consecutiveRunList = []

    for i in range(columnNumber):
        pixelCount = 0                      # Number of pixels of the same value
        columnR = []
        whiteRun = []
        blackRun = []
        col = img[:, i]
        currentType = col[0]
        for j in range(rowNumber):
            if (col[j] == currentType):
                pixelCount += 1
            else:
                columnR.append(pixelCount)
                if (currentType == 0):
                    blackRun.append(pixelCount)
                else:
                    whiteRun.append(pixelCount)

                # Change the run type from W to B or vice versa and initialize count
                currentType = col[j]
                pixelCount = 1

        # Add final run length to encoding
        columnR.append(pixelCount)
        if (currentType == 0):
            blackRun.append(pixelCount)
        else:
            whiteRun.append(pixelCount)

        columnRSum = [sum(columnR[i: i + 2]) for i in range(len(columnR))]
        whiteRunList.extend(whiteRun)
        blackRunList.extend(blackRun)
        consecutiveRunList.extend(columnRSum)

    whiteRuns = Counter(whiteRunList)
    blackRuns = Counter(blackRunList)
    sumRun = Counter(consecutiveRunList)

    lineSpace = whiteRuns.most_common(1)[0][0]
    lineWidth = blackRuns.most_common(1)[0][0]
    widthSum = sumRun.most_common(1)[0][0]

    return lineWidth, lineSpace

# Finding the pixel rows used for staff lines ------------------------------------
def findStaffRows(img, lineWidth, lineSpace, threshold):
    rowNumber = img.shape[0]
    columnNumber = img.shape[1]
    blackPixelChart = []

    # For each row we count the number of black pixels to determine staff lines
    for i in range(rowNumber):
        blackPixelCount = 0
        row = img[i]
        for j in range(len(row)):
            if (row[j] == 0):
                blackPixelCount += 1
        blackPixelChart.append(blackPixelCount)

    # Filter and find the corresponding staff lines by comparing black pixel count to threshold
    staffCount = 5
    staffIndices = []
    staffLength = (staffCount * lineWidth) + ((staffCount - 1) * lineSpace)
    iterationRange = rowNumber - staffLength + 1

    activeRow = 0
    while (activeRow < iterationRange):
        staffLines = [blackPixelChart[j: j + lineWidth] for j in range(activeRow, activeRow + (staffCount - 1) * (lineWidth + lineSpace) + 1, lineWidth + lineSpace)]
        averageNum = sum(sum(staffLines, [])) / (staffCount * lineWidth)
        for line in staffLines:
            if (sum(line) / lineWidth < threshold * columnNumber):
                activeRow += 1
                break
        else:
            staffRowIndices = [list(range(j, j + lineWidth)) for j in range(activeRow, activeRow + (staffCount - 1) * (lineWidth + lineSpace) + 1, lineWidth + lineSpace)]
            staffIndices.append(staffRowIndices)
            activeRow = activeRow + staffLength
    return staffIndices

# Finding the pixel columns used for staff lines ----------------------------------
def findStaffColumns(img, staffVerticalIndices, lineWidth, lineSpace):
    rowNumber = img.shape[0]
    columnNumber = img.shape[1]
    staffExtremum = []

    for i in range(len(staffVerticalIndices)):
        begin = 0
        listBegin = []
        end = columnNumber - 1
        listEnd = []

        # Find the start of staff
        for j in range(columnNumber // 2):
            isolatedStaffRow = img[staffVerticalIndices[i][0][0]:staffVerticalIndices[i][4][
                lineWidth - 1], j]
            blackPixelCount = len(list(filter(lambda x: x == 0, isolatedStaffRow)))
            if (blackPixelCount == 0):
                listBegin.append(j)
        list.sort(listBegin, reverse=True)
        begin = listBegin[0]

        # Find the start of staff
        for j in range(columnNumber // 2, columnNumber):
            isolatedStaffRow = img[staffVerticalIndices[i][0][0]:staffVerticalIndices[i][4][
                lineWidth - 1], j]
            blackPixelCount = len(list(filter(lambda x: x == 0, isolatedStaffRow)))
            if (blackPixelCount == 0):
                listEnd.append(j)
        list.sort(listEnd)
        end = listEnd[0]
        staffEx = (begin, end)
        staffExtremum.append(staffEx)
    return staffExtremum

# Remove the staff lines to show pure notation of music --------------------------
def removeStaffLines(img, staffVerticalIndices):
    noStaffImage = deepcopy(img)
    for staff in staffVerticalIndices:
        for line in staff:
            for row in line:
                # Remove top and bottom line to be sure
                noStaffImage[row - 1, :] = 255
                noStaffImage[row, :] = 255
                noStaffImage[row + 1, :] = 255
    cv2.imwrite('output/noStaffImage.jpg', noStaffImage)
    return noStaffImage

def templateLocation(img, templates, start, stop, threshold):
    locations, scale = match(img, templates, start, stop, threshold)
    imgLocation = []
    for i in range(len(locations)):
        w, h = templates[i].shape[::-1]
        h *= scale
        w *= scale
        imgLocation.append([BoundingBox(pt[0], pt[1], w, h) for pt in zip(*locations[i][::-1])])
    return imgLocation

def mergeBox(boxes, threshold):
    filterBox = []
    while len(boxes) > 0:
        r = boxes.pop(0)
        boxes.sort(key=lambda box: box.distance(r))
        merged = True
        while (merged):
            merged = False
            i = 0
            for _ in range(len(boxes)):
                if r.overlap(boxes[i]) > threshold or boxes[i].overlap(r) > threshold:
                    r = r.merge(boxes.pop(i))
                    merged = True
                elif boxes[i].distance(r) > r.w / 2 + boxes[i].w / 2:
                    break
                else:
                    i += 1
        filterBox.append(r)
    return filterBox