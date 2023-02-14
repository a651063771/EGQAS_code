import math


class Point(object):
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y


class DPCompress(object):
    def __init__(self, pointList, tolerance):
        self.Compressed = list()
        self.pointList = pointList
        self.tolerance = tolerance
        #self.runDP(pointList, tolerance)

    def calc_height(self, point1, point2, point):
        """
        计算point到[point1, point2所在直线]的距离
        点到直线距离：
        A = point2.y - point1.y;
        B = point1.x - point2.x;
        C = point2.x * point1.y - point1.x * point2.y
        Dist = abs((A * point3.X + B * point3.Y + C) / sqrt(A * A + B * B))
        """

        # tops2 = abs(point1.x * point2.y + point2.x * point.y
        #                 + point.x * point1.y - point2.x * point1.y - point.x *
        #                 point2.y - point1.x * point.y)
        # tops = abs(point1.x * point.y + point2.x * point1.y + point.x * point2.y
        #            - point1.x * point2.y - point2.x * point.y - point.x * point1.y
        #            )
        tops = abs(point1[0]* point[1] + point2[0] * point1[1] + point[0] * point2[1]
                   - point1[0] * point2[1] - point2[0] * point[1] - point[0] * point1[1]
                   )
        bottom = math.sqrt(
            math.pow(point2[1] - point1[1], 2) + math.pow(point2[0] - point1[0], 2)
        )

        #height = 100 * tops / bottom
        height = tops / bottom
        #print("height", height)
        return height

    def DouglasPeucker(self, pointList, firsPoint, lastPoint, tolerance):
        """
        计算通过的内容
        DP算法
        :param pointList: 点列表
        :param firsPoint: 第一个点
        :param lastPoint: 最后一个点
        :param tolerance: 容差
        :return:
        """
        maxDistance = 0.0
        indexFarthest = 0
        for i in range(firsPoint, lastPoint):
            distance = self.calc_height(pointList[firsPoint], pointList[lastPoint], pointList[i])
            if (distance > maxDistance):
                maxDistance = distance
                indexFarthest = i
        #    print('max_dis=', maxDistance)

        if maxDistance > tolerance and indexFarthest != 0:
            self.Compressed.append(pointList[indexFarthest])
            self.DouglasPeucker(pointList, firsPoint, indexFarthest, tolerance)
            self.DouglasPeucker(pointList, indexFarthest, lastPoint, tolerance)

    def runDP(self, pointList, tolerance):
        """
        主要运行结果
        :param pointList: Point 列表
        :param tolerance: 值越小，压缩后剩余的越多
        :return:
        """
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


# import pandas as pd
# import numpy as np
# import collections



# def load_data(file_path):
#     columns = ['rid', 'car_id', 'lon', 'lat']
#     df = pd.read_csv(file_path, header=None, names=columns)
#     df_data = df.loc[df['lon'] != np.nan].reset_index().drop(['index'], axis=1)
#
#     car_to_points = collections.defaultdict(list)
#     for i, row in df_data.iterrows():
#         row_id = row['rid']
#         car_id = row['car_id']
#         lon = float(row['lon'])
#         lat = float(row['lat'])
#         pt = Point(row_id, round(lon, 6), round(lat, 6))  # 构造Point对象
#         car_to_points[car_id].append(pt)
#     return car_to_points
#
#
# if __name__ == '__main__':
#     data_file = 'data/trajectory.csv'
#     output_file = 'data/compressed.csv'
#     car_points = load_data(data_file)
#     # print(car_points.keys())
#     with open(output_file, 'w', encoding='utf-8') as fwriter:
#         for car, PointList in car_points.items():
#             points = []
#             dp = DPCompress(PointList, 3.5)
#             points = dp.getCompressed()
#             for p in points:
#                 line = "{},{},{}".format(car, p.x, p.y)
#                 fwriter.write(line)
#                 fwriter.write("\n")
#             print(car, len(PointList), '-->', len(points))
