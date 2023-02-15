import math


class DPCompress(object):
    def __init__(self, pointList, tolerance):
        self.Compressed = list()
        self.pointList = pointList
        self.tolerance = tolerance



    def DouglasPeucker(self, pointList, firsPoint, lastPoint, tolerance):

        maxDistance = 0.0
        indexFarthest = 0
        for i in range(firsPoint, lastPoint):
            distance = self.calc_height(pointList[firsPoint], pointList[lastPoint], pointList[i])
            if (distance > maxDistance):
                maxDistance = distance
                indexFarthest = i


        if maxDistance > tolerance and indexFarthest != 0:
            self.Compressed.append(pointList[indexFarthest])
            self.DouglasPeucker(pointList, firsPoint, indexFarthest, tolerance)
            self.DouglasPeucker(pointList, indexFarthest, lastPoint, tolerance)

    def runDP(self, pointList, tolerance):

        if pointList == None or pointList.__len__() < 3:
            return pointList

        firspoint = 0
        lastPoint = len(pointList) - 1

        self.Compressed.append(pointList[firspoint])
        self.Compressed.append(pointList[lastPoint])

        while (pointList[firspoint] == pointList[lastPoint]):
            lastPoint -= 1
        self.DouglasPeucker(pointList, firspoint, lastPoint, tolerance)
        return sorted(self.Compressed,key=lambda x:(x[0]))
    def getCompressed(self):
        self.Compressed.sort(key=lambda point: int(point.id))
        return self.Compressed



