import cv2, math

class Bar(object):
    def __init__(self, keySign = "c"):
        self.primitives = []

    def getPrimitives(self):
        return self.primitives
    def addPrimitive(self, primitive):
        self.primitives.append(primitive)


class Primitive(object):
    def __init__(self, primitive, duration, box, pitch = -1):
        self.box = box
        self.pitch = pitch
        self.primitive = primitive
        self.duration = duration

    def setDuration(self, duration):
        self.duration = duration
    def getDuration(self):
        return self.duration

    def setPitch(self, pitch):
        self.pitch = pitch
    def getPitch(self):
        return self.pitch

    def getPrimitive(self):
        return self.primitive
    def getBox(self):
        return self.box

class BoundingBox(object):
    def __init__(self, x, y, w, h):
        self.x = x;
        self.y = y;
        self.w = w;
        self.h = h;
        self.middle = self.x + self.w/2, self.y + self.h/2
        self.area = self.w * self.h

    def overlap(self, other):
        overlap_x = max(0, min(self.x + self.w, other.x + other.w) - max(self.x, other.x));
        overlap_y = max(0, min(self.y + self.h, other.y + other.h) - max(self.y, other.y));
        overlap_area = overlap_x * overlap_y
        return overlap_area / self.area

    def distance(self, other):
        dx = self.middle[0] - other.middle[0]
        dy = self.middle[1] - other.middle[1]
        return math.sqrt(dx*dx + dy*dy)

    def merge(self, other):
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        w = max(self.x + self.w, other.x + other.w) - x
        h = max(self.y + self.h, other.y + other.h) - y
        return BoundingBox(x, y, w, h)
        
    def getWidth(self):
        return self.w
    def mulWidth(self, n):
        self.w = self.w * n
    def getHeight(self):
        return self.h
    def mulHeight(self, n):
        self.h = self.h * n

    def draw(self, img, color, thickness):
        pos = ((int)(self.x), (int)(self.y))
        size = ((int)(self.x + self.w/2), (int)(self.y + self.h/2))
        cv2.rectangle(img, pos, size, color, thickness)

    def getCorner(self):
        return self.x, self.y
    def getCenter(self):
        return self.middle

class Staff(object):
    def __init__(self, staffMat, staffLineBox, lineWidth, lineSpace, staffImage, clef="treble", timeSign="44", instrument=-1):
        self.clef = clef
        self.timeSign = timeSign
        self.instrument = instrument
        self.staffLineBox = staffLineBox
        self.lineWidth = lineWidth
        self.lineSpace = lineSpace
        self.img = staffImage
        self.bars = []
        self.one = staffMat[0]
        self.two = staffMat[1]
        self.three = staffMat[2]
        self.four = staffMat[3]
        self.five = staffMat[4]

    def setClef(self, clef):
        self.clef = clef
    def getClef(self):
        return self.clef

    def setTimeSignature(self, time):
        self.timeSign = time
    def getTimeSignature(self):
        return self.timeSign

    def setInstrument(self, instrument):
        self.instrument = instrument

    def addBar(self, bar):
        self.bars.append(bar)
    def getBars(self):
        return self.bars

    def getBox(self):
        return self.staffLineBox
    def getImage(self):
        return self.img
    def getLineWidth(self):
        return self.lineWidth
    def getLineSpacing(self):
        return self.lineSpace


    def getPitch(self, noteCenterY):
        noteName = ["C", "D", "E", "F", "G", "A", "B"]
        clefOrder = {
            "bass": [("A3", "G3", "F3", "E3", "D3", "C3", "B2", "A2", "G2"), (3,5), (2,4)],
            "treble": [("F5", "E5", "D5", "C5", "B4", "A4", "G4", "F4", "E4"), (5,3), (4,2)]
        }

        if (noteCenterY in list(range(self.one[0] - 3, self.one[-1] + 3))):
            return clefOrder[self.clef][0][0]
        elif (noteCenterY in list(range(self.one[-1] + 3, self.two[0] - 3))):
            return clefOrder[self.clef][0][1]
        elif (noteCenterY in list(range(self.two[0] - 3, self.two[-1] + 3))):
            return clefOrder[self.clef][0][2]
        elif (noteCenterY in list(range(self.two[-1] + 3, self.three[0] - 3))):
            return clefOrder[self.clef][0][3]
        elif (noteCenterY in list(range(self.three[0] - 3, self.three[-1] + 3))):
            return clefOrder[self.clef][0][4]
        elif (noteCenterY in list(range(self.three[-1] + 3, self.four[0] - 3))):
            return clefOrder[self.clef][0][5]
        elif (noteCenterY in list(range(self.four[0] - 3, self.four[-1] + 3))):
            return clefOrder[self.clef][0][6]
        elif (noteCenterY in list(range(self.four[-1] + 3, self.five[0] - 3))):
            return clefOrder[self.clef][0][7]
        elif (noteCenterY in list(range(self.five[0] - 3, self.five[-1] + 3))):
            return clefOrder[self.clef][0][8]
        else:
            if (noteCenterY < self.one[0] - 3):
                belowLine = self.one
                curLine = [pixel - self.lineSpace for pixel in self.one]
                octave = clefOrder[self.clef][1][0]
                indexOfNote = clefOrder[self.clef][1][1]

                while (curLine[0] > 0):
                    if (noteCenterY in curLine):
                        octave = octave + 1 if (indexOfNote + 2 >= 7) else octave
                        indexOfNote = (indexOfNote + 2) % 7
                        return noteName[indexOfNote] + str(octave)
                    elif (noteCenterY in range(curLine[-1] + 1, belowLine[0])):
                        octave = octave + 1 if (indexOfNote + 1 >= 7) else octave
                        indexOfNote = (indexOfNote + 1) % 7
                        return noteName[indexOfNote] + str(octave)
                    else:
                        octave = octave + 1 if (indexOfNote + 2 >= 7) else octave
                        indexOfNote = (indexOfNote + 2) % 7
                        belowLine = curLine.copy()
                        curLine = [pixel - self.lineSpace for pixel in curLine]
            elif (noteCenterY > self.five[-1] + 3):
                aboveLine = self.five
                curLine = [pixel + self.lineSpace for pixel in self.five]
                octave = clefOrder[self.clef][2][0]
                indexOfNote = clefOrder[self.clef][2][1]

                while (curLine[-1] < self.img.shape[0]):
                    if (noteCenterY in curLine):
                        octave = octave - 1 if (indexOfNote - 2 <= 7) else octave
                        indexOfNote = (indexOfNote - 2) % 7
                        return noteName[indexOfNote] + str(octave)
                    elif (noteCenterY in range(aboveLine[-1] + 1, curLine[0])):
                        octave = octave - 1 if (indexOfNote - 1 >= 7) else octave
                        indexOfNote = (indexOfNote - 1) % 7
                        return noteName[indexOfNote] + str(octave)
                    else:
                        octave = octave - 1 if (indexOfNote - 2 <= 7) else octave
                        indexOfNote = (indexOfNote - 2) % 7
                        aboveLine = curLine.copy()
                        curLine = [pixel + self.lineSpace for pixel in curLine]