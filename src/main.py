"""
Optical Music Recognition and Playback
Signals and Systems 25742
Dr. Babak Khalaj

Student: Amirmahdi Soleimanifar
ID: 98101747
Email: asoleix@gmail.com
"""
# Import dependencies and files ------------------------------------------------
import sys
import cv2                              # Opencv-python library
import numpy as np
from PIL import Image                   # Python imaging library
from functions import *
from midiutil.MidiFile import MIDIFile
from music21 import converter, instrument
from pdf2image import convert_from_path

# Templates upper and lower bound and thresholds ------------------------------
clefLow, clefUp, clefThreshold = 50, 150, 0.18
timeLow, timeUp, timeThreshold = 50, 150, 0.65
sharpLow, sharpUp, sharpThreshold = 50, 150, 0.65
flatLow, flatUp, flatThreshold = 50, 150, 0.76
quarterNoteLow, quarterNoteUp, quarterNoteThreshold = 50, 150, 0.70
halfNoteLow, halfNoteUp, halfNoteThreshold = 50, 150, 0.70
wholeNoteLow, wholeNoteUp, wholeNoteThreshold = 50, 150, 0.70
eighthRestLow, eighthRestUp, eighthRestThreshold = 50, 150, 0.75
quarterRestLower, quarterRestUp, quarterRestThreshold = 50, 150, 0.70
halfRestLow, halfRestUp, halfRestThreshold = 50, 150, 0.80
wholeRestLow, wholeRestUp, wholeRestThreshold = 50, 150, 0.80
eighthFlagLow, eighthFlagUp, eighthFlagThreshold = 50, 150, 0.8
barLow, barUp, barThreshold = 50, 150, 0.85

# Mapping the recognized note to the MIDI corresponding number -------------------
convertToPitch = {
    108: "C8", 107: "B7", 106: "A#7", 105: "A7", 104: "G#7", 103: "G7", 102: "F#7", 101: "F7", 100: "E7", 99: "D#7", 98: "D7", 97: "C#7", 96: "C7", 95: "B6", 94: "A#6", 93: "A6",
    92: "G#6", 91: "G6", 90: "F#6", 89: "F6", 88: "E6", 87: "D#6", 86: "D6", 85: "C#6", 84: "C6", 83: "B5", 82: "A#5", 81: "A5", 80: "G#5", 79: "G5", 78: "F#5", 77: "F5", 76: "E5",
    75: "D#5", 74: "D5", 73: "C#5", 72: "C5", 71: "B4", 70: "A#4", 69: "A4", 68: "G#4", 67: "G4", 66: "F#4", 65: "F4", 64: "E4", 63: "D#4", 62: "D4", 61: "C#4", 60: "C4", 59: "B3",
    58: "A#3", 57: "A3", 56: "G#3", 55: "G3", 54: "F#3", 53: "F3", 52: "E3", 51: "D#3", 50: "D3", 49: "C#3", 48: "C3", 47: "B2", 46: "A#2", 45: "A2", 44: "G#2", 43: "G2", 42: "F#2",
    41: "F2", 40: "E2", 39: "D#2", 38: "D2", 37: "C#2", 36: "C2", 35: "B1", 34: "A#1", 33: "A1", 32: "G#1", 31: "G1", 30: "F#1", 29: "F1", 28: "E1", 27: "D#1", 26: "D1", 25: "C#1",
    24: "C1", 23: "B0", 22: "A#0", 21: "A0"
}

convertToMIDI = {
    "C8": 108, "B7": 107, "Bb7": 106, "A#7": 106, "A7": 105, "Ab7": 104, "G#7": 104, "G7": 103, "Gb7": 102, "F#7": 102, "F7": 101, "E7": 100, "Eb7": 99, "D#7": 99, "D7": 98, 
    "Db7": 97, "C#7": 97, "C7": 96, "B6": 95, "Bb6": 94, "A#6": 94, "A6": 93, "Ab6": 92, "G#6": 92, "G6": 91, "Gb6": 90, "F#6": 90, "F6": 89, "E6": 88, "Eb6": 87, "D#6": 87,
    "D6": 86, "Db6": 85, "C#6": 85, "C6": 84, "B5": 83, "Bb5": 82, "A#5": 82, "A5": 81, "Ab5": 80, "G#5": 80, "G5": 79, "Gb5": 78, "F#5": 78, "F5": 77, "E5": 76, "Eb5": 75,
    "D#5": 75, "D5": 74, "Db5": 73, "C#5": 73, "C5": 72, "B4": 71, "Bb4": 70, "A#4": 70, "A4": 69, "Ab4": 68, "G#4": 68, "G4": 67, "Gb4": 66, "F#4": 66, "F4": 65, "E4": 64,
    "Eb4": 63, "D#4": 63, "D4": 62, "Db4": 61, "C#4": 61, "C4": 60, "B3": 59, "Bb3": 58, "A#3": 58, "A3": 57, "Ab3": 56, "G#3": 56, "G3": 55, "Gb3": 54, "F#3": 54, "F3": 53,
    "E3": 52, "Eb3": 51, "D#3": 51, "D3": 50, "Db3": 49, "C#3": 49, "C3": 48, "B2": 47, "Bb2": 46, "A#2": 46, "A2": 45, "Ab2": 44, "G#2": 44, "G2": 43, "Gb2": 42, "F#2": 42,
    "F2": 41, "E2": 40, "Eb2": 39, "D#2": 39, "D2": 38, "Db2": 37, "C#2": 37, "C2": 36, "B1": 35, "Bb1": 34, "A#1": 34, "A1": 33,  "Ab1": 32, "G#1": 32, "G1": 31, "Gb1": 30,
    "F#1": 30, "F1": 29, "E1": 28, "Eb1": 27, "D#1": 27, "D1": 26, "Db1": 25, "C#1": 25, "C1": 24, "B0": 23, "Bb0": 22, "A#0": 22, "A0": 21
}

keySigns = {
    "sharp": ["", "F", "FC", "FCG", "FCGD", "FCGDA", "FCGDAE", "FCGDAEB"],
    "flat": ["", "B", "BE", "BEA", "BEAD", "BEADG", "BEADGC", "BEADGCF"]
}

# Determining address of templates ------------------------------------------------

barPath = ["asset/barline/BL1.jpg", "asset/barline/BL2.jpg", "asset/barline/BL3.jpg", "asset/barline/BL4.jpg"]
barImg = [cv2.imread(barline, 0) for barline in barPath]

characterPath = {
    "sharp": ["asset/character/sharp-line.jpg", "asset/character/sharp-space.jpg"],
    "flat": ["asset/character/flat-line.jpg", "asset/character/flat-space.jpg"]
}
flatImg = [cv2.imread(flatFile, 0) for flatFile in characterPath["flat"]]
sharpImg = [cv2.imread(sharpFile, 0) for sharpFile in characterPath["sharp"]]

clefPath = {
    "treble": ["asset/clef/treble1.jpg", "asset/clef/treble2.jpg"],
    "bass": ["asset/clef/bass.jpg"]
}
clefImg = {
    "treble": [cv2.imread(treble, 0) for treble in clefPath["treble"]],
    "bass": [cv2.imread(bass, 0) for bass in clefPath["bass"]]
}

notePath = {
    "whole": [ "asset/note/whole-space.png", "asset/note/whole-note-line.png", "asset/note/whole-line.png", "asset/note/whole-note-space.png"],
    "half": ["asset/note/half-space.png", "asset/note/half-note-line.png", "asset/note/half-line.png", "asset/note/half-note-space.png"],
    "quarter": ["asset/note/quarter.png", "asset/note/solid-note.png"]
}
wholeNoteImg = [cv2.imread(whole, 0) for whole in notePath['whole']]
halfNoteImg = [cv2.imread(half, 0) for half in notePath["half"]]
quarterNoteImg = [cv2.imread(quarter, 0) for quarter in notePath["quarter"]]

restPath = {
    "whole": ["asset/rest/whole-rest.jpg"],
    "half": ["asset/rest/half-rest1.jpg", "asset/rest/half-rest2.jpg"],
    "quarter": ["asset/rest/quarter-rest.jpg"],
    "eighth": ["asset/rest/eighth-rest.jpg"]
}
wholeRestImg = [cv2.imread(whole, 0) for whole in restPath['whole']]
halfRestImg = [cv2.imread(half, 0) for half in restPath["half"]]
quarterRestImg = [cv2.imread(quarter, 0) for quarter in restPath["quarter"]]
eighthRestImg = [cv2.imread(eighth, 0) for eighth in restPath["eighth"]]

flagPath = ["asset/flag/EF1.jpg", "asset/flag/EF2.jpg","asset/flag/EF3.jpg",
            "asset/flag/EF4.jpg", "asset/flag/EF5.jpg", "asset/flag/EF6.jpg"]
flagImg = [cv2.imread(flag, 0) for flag in flagPath]

timeImg = {
    "common": [cv2.imread(time, 0) for time in ["asset/time/common.jpg"]],
    "44": [cv2.imread(time, 0) for time in ["asset/time/44.jpg"]],
    "34": [cv2.imread(time, 0) for time in ["asset/time/34.jpg"]],
    "24": [cv2.imread(time, 0) for time in ["asset/time/24.jpg"]],
    "68": [cv2.imread(time, 0) for time in ["asset/time/68.jpg"]]
}

# The command line starts the execution -----------------------------------------
if __name__ == "__main__":

    pdfFile = sys.argv[1:][0]
    imageFile = convert_from_path(pdfFile)
    for image in imageFile:
        image.save('output/musicSheet.jpg', 'JPEG')
    img = cv2.imread('output/musicSheet.jpg', 0)
    img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    # Using binarization to have only black and white colors in the picture to enhance recognition
    # The method used for binarization is Otsu's thresholding
    retval, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite('output/binarizedImage.jpg', img)
    lineWidth, lineSpace = lengthReference(img)
    print("Staff line width is", lineWidth, "and staff line spacing is", lineSpace)

    # To detect staff lines we use a histogram determining the number of black pixels in each row
    # The top 5 rows are selected as staff lines, with a little compromise for angle of lines.
    staffVerticalIndices = findStaffRows(img, lineWidth, lineSpace, 0.1)
    print("The sheet contains", len(staffVerticalIndices), "sets of staff lines")
    
    # Searching for columns that has no black pixels with largest index
    staffHorizontalIndices = findStaffColumns(img, staffVerticalIndices, lineWidth, lineSpace)
    removeStaffLines(img, staffVerticalIndices)

    # Show the detected staffs in an image saved in the output -------------------
    staffs = []
    if(len(staffVerticalIndices)>1):
        distBetweenStaffs = (staffVerticalIndices[1][0][0] - staffVerticalIndices[0][4][lineWidth - 1])//2
    else:
        distBetweenStaffs = 0
    for i in range(len(staffVerticalIndices)):
        x = staffHorizontalIndices[i][0]
        y = staffVerticalIndices[i][0][0]
        width = staffHorizontalIndices[i][1] - x
        height = staffVerticalIndices[i][4][lineWidth - 1] - y
        staffLineBox = BoundingBox(x, y, width, height)
        # Create Cropped Staff Image and normalize staff line numbers
        staffImage = img[max(0, y - distBetweenStaffs): min(y+ height + distBetweenStaffs, img.shape[0] - 1), x:x+width]
        pixel = distBetweenStaffs
        normalizedVerticalIndices = []
        for j in range(5):
            line = []
            for k in range(lineWidth):
                line.append(pixel)
                pixel += 1
            normalizedVerticalIndices.append(line)
            pixel += lineSpace + 1
        staff = Staff(normalizedVerticalIndices, staffLineBox, lineWidth, lineSpace, staffImage)
        staffs.append(staff)

    staffLineDetectionImage = img.copy()
    staffLineDetectionImage = cv2.cvtColor(staffLineDetectionImage, cv2.COLOR_GRAY2RGB)
    color = (188, 86, 214)
    boxThick = 4
    for staff in staffs:
        box = staff.getBox()
        box.mulWidth(2)
        box.mulHeight(2)
        box.draw(staffLineDetectionImage, color, boxThick)
        box.mulWidth(0.5)
        box.mulHeight(0.5)
        x = int(box.getCorner()[0] + (box.getWidth() // 2))
        y = int(box.getCorner()[1] + box.getHeight() + 35)
    cv2.imwrite('output/staffDetection.jpg', staffLineDetectionImage)

    # The first step is to recognize all primitives in each staff and then structure and recognize music
    staffImagesColor = []
    for i in range(len(staffs)):
        color = (0, 0, 255)
        boxThick = 2
        staffImage = staffs[i].getImage()
        staffImgColor = staffImage.copy()
        staffImgColor = cv2.cvtColor(staffImgColor, cv2.COLOR_GRAY2RGB)

        # Determining the time on each staff
        color = (0, 122, 255)
        boxThick = 3
        for time in timeImg:
            timeBox = templateLocation(staffImage, timeImg[time], timeLow, timeUp, timeThreshold)
            timeBox = mergeBox([j for i in timeBox for j in i], 0.5)
            if (len(timeBox) == 1):
                print("Time Signature is", time)
                staffs[i].setTimeSignature(time)
                for boxes in timeBox:
                    boxes.draw(staffImgColor, color, boxThick)
                    x = int(boxes.getCorner()[0] - (boxes.getWidth() // 2))
                    y = int(boxes.getCorner()[1] + boxes.getHeight() + 20)
                    cv2.putText(staffImgColor, "{} time".format(time), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.1, color)
                break
            elif (len(timeBox) == 0 and i > 0):
                previousTime = staffs[i-1].getTimeSignature()
                staffs[i].setTimeSignature(previousTime)
                break
        else:
            print("No time signature for staff", i + 1)

        staffImagesColor.append(staffImgColor)

        # Determining the clef of each staff
        for clef in clefImg:
            clefBox = templateLocation(staffImage, clefImg[clef], clefLow, clefUp, clefThreshold)
            clefBox = mergeBox([j for i in clefBox for j in i], 0.5)
            if (len(clefBox) == 1):
                print("Clef on staff", i+1 ,"is", clef)
                staffs[i].setClef(clef)
                clefBoxImg = staffs[i].getImage()
                clefBoxImg = clefBoxImg.copy()
                for boxes in clefBox:
                    boxes.mulWidth(0.0667)
                    boxes.draw(staffImgColor, color, boxThick)
                    boxes.mulWidth(15)
                    x = int(boxes.getCorner()[0] + (boxes.getWidth() // 2))
                    y = int(boxes.getCorner()[1] + boxes.getHeight() + 10)
                    cv2.putText(staffImgColor, "{} clef".format(clef), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.1, color)
                break
        else:
            print("No clef on staff", i+1)

    # Finding the related primitives ----------------------------------------------
    for i in range(len(staffs)):
        staffPrimitive = []
        staffImage = staffs[i].getImage()
        staffImgColor = staffImagesColor[i]
        color = (76, 217, 100)
        boxThick = 2

        print("Matching bar line templates on staff", i+1, "...")
        barlineBox = templateLocation(staffImage, barImg, barLow, barUp, barThreshold)
        barlineBox = mergeBox([j for i in barlineBox for j in i], 0.5)
        for box in barlineBox:
            box.draw(staffImgColor, color, boxThick)
            text = "line"
            font = cv2.FONT_HERSHEY_PLAIN
            textsize = cv2.getTextSize(text, font, fontScale=0.9, thickness=1)[0]
            x = int(box.getCorner()[0] - (textsize[0] // 2))
            y = int(box.getCorner()[1] + box.getHeight() + 20)
            cv2.putText(staffImgColor, text, (x, y), font, fontScale=0.9, color=color, thickness=1)
            line = Primitive("line", 0, box)
            staffPrimitive.append(line)

        print('Matching sharp and flat characters on staff', i+1, '...')
        flaxBox = templateLocation(staffImage, flatImg, flatLow, flatUp, flatThreshold)
        flaxBox = mergeBox([j for i in flaxBox for j in i], 0.5)
        for box in flaxBox:
            box.draw(staffImgColor, color, boxThick)
            text = "flat"
            font = cv2.FONT_HERSHEY_PLAIN
            textsize = cv2.getTextSize(text, font, fontScale=0.9, thickness=1)[0]
            x = int(box.getCorner()[0] - (textsize[0] // 2))
            y = int(box.getCorner()[1] + box.getHeight() + 20)
            cv2.putText(staffImgColor, text, (x, y), font, fontScale=0.9, color=color, thickness=1)
            flat = Primitive("flat", 0, box)
            staffPrimitive.append(flat)

        sharpBox = templateLocation(staffImage, sharpImg, sharpLow, sharpUp, sharpThreshold)
        sharpBox = mergeBox([j for i in sharpBox for j in i], 0.5)
        for box in sharpBox:
            box.draw(staffImgColor, color, boxThick)
            text = "sharp"
            font = cv2.FONT_HERSHEY_PLAIN
            textsize = cv2.getTextSize(text, font, fontScale=0.9, thickness=1)[0]
            x = int(box.getCorner()[0] - (textsize[0] // 2))
            y = int(box.getCorner()[1] + box.getHeight() + 20)
            cv2.putText(staffImgColor, text, (x, y), font, fontScale=0.9, color=color, thickness=1)
            sharp = Primitive("sharp", 0, box)
            staffPrimitive.append(sharp)

        print("Matching note templates with notation on staff", i+1, "...")
        wholeBox = templateLocation(staffImage, wholeNoteImg, wholeNoteLow, wholeNoteUp, wholeNoteThreshold)
        wholeBox = mergeBox([j for i in wholeBox for j in i], 0.5)
        for box in wholeBox:
            box.draw(staffImgColor, color, boxThick)
            text = "1 note"
            font = cv2.FONT_HERSHEY_PLAIN
            textsize = cv2.getTextSize(text, font, fontScale=0.9, thickness=1)[0]
            x = int(box.getCorner()[0] - (textsize[0] // 2))
            y = int(box.getCorner()[1] + box.getHeight() + 20)
            cv2.putText(staffImgColor, text, (x, y), font, fontScale=0.9, color=color, thickness=1)
            pitch = staffs[i].getPitch(round(box.getCenter()[1]))
            whole = Primitive("note", 4, box, pitch)
            staffPrimitive.append(whole)

        halfBox = templateLocation(staffImage, halfNoteImg, halfNoteLow, halfNoteUp, halfNoteThreshold)
        halfBox = mergeBox([j for i in halfBox for j in i], 0.5)
        for box in halfBox:
            box.draw(staffImgColor, color, boxThick)
            text = "1/2 note"
            font = cv2.FONT_HERSHEY_PLAIN
            textsize = cv2.getTextSize(text, font, fontScale=0.9, thickness=1)[0]
            x = int(box.getCorner()[0] - (textsize[0] // 2))
            y = int(box.getCorner()[1] + box.getHeight() + 20)
            cv2.putText(staffImgColor, text, (x, y), font, fontScale=0.9, color=color, thickness=1)
            pitch = staffs[i].getPitch(round(box.getCenter()[1]))
            half = Primitive("note", 2, box, pitch)
            staffPrimitive.append(half)

        quarterBox = templateLocation(staffImage, quarterNoteImg, quarterNoteLow, quarterNoteUp, quarterNoteThreshold)
        quarterBox = mergeBox([j for i in quarterBox for j in i], 0.5)
        for box in quarterBox:
            box.draw(staffImgColor, color, boxThick)
            text = "1/4 note"
            font = cv2.FONT_HERSHEY_PLAIN
            textsize = cv2.getTextSize(text, font, fontScale=0.9, thickness=1)[0]
            x = int(box.getCorner()[0] - (textsize[0] // 2))
            y = int(box.getCorner()[1] + box.getHeight() + 20)
            cv2.putText(staffImgColor, text, (x, y), font, fontScale=0.9, color=color, thickness=1)
            pitch = staffs[i].getPitch(round(box.getCenter()[1]))
            quarter = Primitive("note", 1, box, pitch)
            staffPrimitive.append(quarter)

        eighthBox = templateLocation(staffImage, eighthRestImg, eighthRestLow, eighthRestUp, eighthRestThreshold)
        eighthBox = mergeBox([j for i in eighthBox for j in i], 0.5)
        for box in eighthBox:
            box.draw(staffImgColor, color, boxThick)
            text = "1/8 rest"
            font = cv2.FONT_HERSHEY_PLAIN
            textsize = cv2.getTextSize(text, font, fontScale=0.9, thickness=1)[0]
            x = int(box.getCorner()[0] - (textsize[0] // 2))
            y = int(box.getCorner()[1] + box.getHeight() + 20)
            cv2.putText(staffImgColor, text, (x, y), font, fontScale=0.9, color=color, thickness=1)
            eighth = Primitive("rest", 0.5, box)
            staffPrimitive.append(eighth)

        print("Matching rest templates on staff", i+1, "...")
        wholeBox = templateLocation(staffImage, wholeRestImg, wholeRestLow, wholeRestUp, wholeRestThreshold)
        wholeBox = mergeBox([j for i in wholeBox for j in i], 0.5)
        for box in wholeBox:
            box.draw(staffImgColor, color, boxThick)
            text = "1 rest"
            font = cv2.FONT_HERSHEY_PLAIN
            textsize = cv2.getTextSize(text, font, fontScale=0.9, thickness=1)[0]
            x = int(box.getCorner()[0] - (textsize[0] // 2))
            y = int(box.getCorner()[1] + box.getHeight() + 20)
            cv2.putText(staffImgColor, text, (x, y), font, fontScale=0.9, color=color, thickness=1)
            whole = Primitive("rest", 4, box)
            staffPrimitive.append(whole)

        halfBox = templateLocation(staffImage, halfRestImg, halfRestLow, halfRestUp, halfRestThreshold)
        halfBox = mergeBox([j for i in halfBox for j in i], 0.5)
        for box in halfBox:
            box.draw(staffImgColor, color, boxThick)
            text = "1/2 rest"
            font = cv2.FONT_HERSHEY_PLAIN
            textsize = cv2.getTextSize(text, font, fontScale=0.9, thickness=1)[0]
            x = int(box.getCorner()[0] - (textsize[0] // 2))
            y = int(box.getCorner()[1] + box.getHeight() + 20)
            cv2.putText(staffImgColor, text, (x, y), font, fontScale=0.9, color=color, thickness=1)
            half = Primitive("rest", 2, box)
            staffPrimitive.append(half)

        quarterBox = templateLocation(staffImage, quarterRestImg, quarterRestLower, quarterRestUp, quarterRestThreshold)
        quarterBox = mergeBox([j for i in quarterBox for j in i], 0.5)
        for box in quarterBox:
            box.draw(staffImgColor, color, boxThick)
            text = "1/4 rest"
            font = cv2.FONT_HERSHEY_PLAIN
            textsize = cv2.getTextSize(text, font, fontScale=0.9, thickness=1)[0]
            x = int(box.getCorner()[0] - (textsize[0] // 2))
            y = int(box.getCorner()[1] + box.getHeight() + 20)
            cv2.putText(staffImgColor, text, (x, y), font, fontScale=0.9, color=color, thickness=1)
            quarter = Primitive("rest", 1, box)
            staffPrimitive.append(quarter)

        print("Matching eighth note flag templates on staff", i+1, "...")
        flagBox = templateLocation(staffImage, flagImg, eighthFlagLow, eighthFlagUp, eighthFlagThreshold)
        flagBox = mergeBox([j for i in flagBox for j in i], 0.5)
        for box in flagBox:
            box.draw(staffImgColor, color, boxThick)
            text = "1/8 flag"
            font = cv2.FONT_HERSHEY_PLAIN
            textsize = cv2.getTextSize(text, font, fontScale=0.9, thickness=1)[0]
            x = int(box.getCorner()[0] - (textsize[0] // 2))
            y = int(box.getCorner()[1] + box.getHeight() + 20)
            cv2.putText(staffImgColor, text, (x, y), font, fontScale=0.9, color=color, thickness=1)
            flag = Primitive("eighth_flag", 0, box)
            staffPrimitive.append(flag)
        cv2.imwrite("output/staff{}Primitives.jpg".format(i+1), staffImgColor)


        # Sort the found primitives from left to right to form the complete sheet
        staffPrimitive.sort(key=lambda primitive: primitive.getBox().getCenter())
        eighthFlagIndices = []
        for j in range(len(staffPrimitive)):
            if (staffPrimitive[j].getPrimitive() == "eighth_flag"):
                eighthFlagIndices.append(j)
            if (staffPrimitive[j].getPrimitive() == "note"):
                if(j != len(staffPrimitive) - 1):
                    print(staffPrimitive[j].getPitch(), end=", ")
                else:
                    print(staffPrimitive[j].getPitch())
            else:
                if(j != len(staffPrimitive) - 1):
                    print(staffPrimitive[j].getPrimitive(), end=", ")
                else:
                    print(staffPrimitive[j].getPrimitive())
        for j in eighthFlagIndices:
            distances = []
            distance = staffPrimitive[j].getBox().distance(staffPrimitive[j-1].getBox())
            distances.append(distance)
            if (j + 1 < len(staffPrimitive)):
                distance = staffPrimitive[j].getBox().distance(staffPrimitive[j+1].getBox())
                distances.append(distance)
            if (distances[1] and distances[0] > distances[1]):
                staffPrimitive[j+1].setDuration(0.5)
            else:
                staffPrimitive[j-1].setDuration(0.5)
            del staffPrimitive[j]

        # If the number of black pixels in the center row of two notes is greater than 5*lineWidth, then notes are beamed.
        for j in range(len(staffPrimitive)):
            if (j+1 < len(staffPrimitive)
                and staffPrimitive[j].getPrimitive() == "note"
                and staffPrimitive[j+1].getPrimitive() == "note"
                and (staffPrimitive[j].getDuration() == 1 or staffPrimitive[j].getDuration() == 0.5)
                and staffPrimitive[j+1].getDuration() == 1):
                note1X = staffPrimitive[j].getBox().getCenter()[0]
                note2X = staffPrimitive[j+1].getBox().getCenter()[0]
                blackPixelCount = 5 * staffs[i].getLineWidth()
                centerCol = (note2X - note1X) // 2
                midCol = staffImage[:, int(note1X + centerCol)]
                blackCountMiddle = len(np.where(midCol == 0)[0])
                if (blackCountMiddle > blackPixelCount):
                    staffPrimitive[j].setDuration(0.5)
                    staffPrimitive[j+1].setDuration(0.5)

        print("Applying changes due to key signature note values...")
        j = 0
        sharpsCount = 0
        flatsCount = 0
        if(len(staffPrimitive) > 0):
            while (staffPrimitive[j].getDuration() == 0):
                character = staffPrimitive[j].getPrimitive()
                if (character == "sharp"):
                    sharpsCount += 1
                    j += 1
                elif (character == "flat"):
                    flatsCount += 1
                    j += 1

        # Determine whether the character is independent or belongs to a note
        if (j != 0):
            centerXChar = staffPrimitive[j-1].getBox().getCenter()[0]
            maxXOffset = staffPrimitive[j].getBox().getCenter()[0] - staffPrimitive[j].getBox().getWidth()
            typeChar = staffPrimitive[j-1].getPrimitive()
            if (centerXChar > maxXOffset):
                sharpsCount = sharpsCount - 1 if typeChar == "sharp" else sharpsCount
                flatsCount = flatsCount - 1 if typeChar == "flat" else flatsCount

            characterNoteList = []
            if (typeChar == "sharp"):
                characterNoteList = keySigns[typeChar][sharpsCount]
                staffPrimitive = staffPrimitive[sharpsCount:]
            else:
                characterNoteList = keySigns[typeChar][flatsCount]
                staffPrimitive = staffPrimitive[flatsCount:]

            for primitive in staffPrimitive:
                type = primitive.getPrimitive()
                note = primitive.getPitch()
                if (type == "note" and note[0] in characterNoteList):
                    rawNote = convertToPitch[convertToMIDI[note] + 1] if typeChar == "sharp" else convertToPitch[convertToMIDI[note] - 1]
                    primitive.setPitch(rawNote)
                if (primitive.getPrimitive() == "note"):
                    print(primitive.getPitch(), end=", ")
                else:
                    print(primitive.getPrimitive(), end=", ")

        # After finding the sharp and flat characters, we associate them with actual notes
        removedPrimitives = []
        for j in range(len(staffPrimitive)):
            typeChar = staffPrimitive[j].getPrimitive()

            if (typeChar == "flat" or typeChar == "sharp"):
                maxXOffset = staffPrimitive[j+1].getBox().getCenter()[0] - staffPrimitive[j+1].getBox().getWidth()
                centerXChar = staffPrimitive[j].getBox().getCenter()[0]
                primitiveType = staffPrimitive[j+1].getPrimitive()
                if (centerXChar > maxXOffset and primitiveType == "note"):
                    note = staffPrimitive[j+1].getPitch()
                    rawNote = convertToPitch[convertToMIDI[note] + 1] if typeChar == "sharp" else convertToPitch[convertToMIDI[note] - 1]
                    staffPrimitive[j+1].setPitch(rawNote)
                    removedPrimitives.append(i)
        for j in removedPrimitives:
            del staffPrimitive[j]

        for j in range(len(staffPrimitive)):
            if (staffPrimitive[j].getPrimitive() == "note"):
                if(j != len(staffPrimitive) - 1):
                    print(staffPrimitive[j].getPitch(), end=", ")
                else:
                    print(staffPrimitive[j].getPitch())
            else:
                if(j != len(staffPrimitive) - 1):
                    print(staffPrimitive[j].getPrimitive(), end=", ")
                else:
                    print(staffPrimitive[j].getPrimitive())

        print("Final creation of staff", i+1, "...")
        bar = Bar()
        while (len(staffPrimitive) > 0):
            primitive = staffPrimitive.pop(0)
            if (primitive.getPrimitive() != "line"):
                bar.addPrimitive(primitive)
            else:
                staffs[i].addBar(bar)
                bar = Bar()
        staffs[i].addBar(bar)

    # Create and show the MIDI file

    midi = MIDIFile(1)
    track = 0
    time = 0
    channel = 0
    volume = 100
    midi.addTrackName(track, time, "Track")
    midi.addTempo(track, time, 110)
    for i in range(len(staffs)):
        print("*** Notes on staff", format(i+1), "***")
        bars = staffs[i].getBars()
        for j in range(len(bars)):
            print("Bar", format(j + 1))
            primitives = bars[j].getPrimitives()
            for k in range(len(primitives)):
                duration = primitives[k].getDuration()
                if (primitives[k].getPitch() == None):
                    continue
                if (primitives[k].getPrimitive() == "note"):
                    pitch = convertToMIDI[primitives[k].getPitch()]
                    midi.addNote(track, channel, pitch, time, duration, volume)
                print(primitives[k].getPrimitive(), primitives[k].getPitch(), primitives[k].getDuration())
                time += duration
            print("------")

    # Creating the output -------------------------------------
    print("Creating the output MIDI file")
    binfile = open("output/output-piano.mid", 'wb')
    midi.writeFile(binfile)
    binfile.close()
    violinSound = converter.parse('output/output-piano.mid')
    for p in violinSound.parts:
        p.insert(0, instrument.Violin())
    violinSound.write('midi', "output/output-violin.mid")