from PIL import Image
import cv2 as cv
import numpy as np
from scipy.spatial import ConvexHull
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
import ConverxHull

# 计算凸包坐标
def lineHepler(point1,point2):
  if(point1[0] - point2[0] != 0):
    m = (point1[1]-point2[1])/(point1[0]-point2[0])   #k
    c = point1[1] - m*point1[0]     #b
    return [m,-1,c]        #返回A，B和C
  else:
    return [1,0,point1[0]]     #当两点之间的线为平行于y轴的线时

def converxHull(points):
  solution = []
  if len(points) <= 3:
    return points         #如果点集中的点少于三个，
  points.sort(key=lambda x:x[0])  #按x坐标排序
  left_most = points[0]
  right_most = points[len(points)-1]
  solution.append(left_most)
  solution.append(right_most)
  helper(points,left_most,right_most,True)
  helper(points,left_most,right_most,False)
  return solution

def helper(points,left_most,right_most,upBool):
  solution = []
  if len(points) <= 1:        #如果点数为1或者为0
    return
  l = lineHepler(left_most,right_most)
  if upBool:
    up = []
    maxdistance = 0
    max_point = ()
    for point in points:
      distance = 0-(l[0]*point[0]+l[1]*point[1]+l[2])/sqrt(l[0]*l[0]+l[1]*l[1])
      if distance > 0:
        up.append(point)
        if distance > maxdistance:
          maxdistance = distance
          max_point = point
    if max_point != ():
      solution.append(max_point)
    helper(up,left_most,max_point,True)   #递归左上包
    helper(up,max_point,right_most,True)   #递归右上包
  else:
    down = []
    min_distance = 0
    min_point = ()
    for point in points:
      distance = 0-(l[0]*point[0]+l[1]*point[1]+l[2])/sqrt(l[0]*l[0]+l[1]*l[1])
      if distance < 0:
        down.append(point)
        if distance<min_distance:
          min_distance = distance
          min_point = point
    if min_point != ():
      solution.append(min_point)
    helper(down,left_most,min_point,False)    #递归左下包
    helper(down,min_point,right_most,False)    #递归右下包

# 计算距离
def Distance(X,Y):
  return pow(pow(X,2) + pow(Y,2), 0.5)

# 计算角度并排序
def CoorSort(XYlist):
  arclist = []
  dislist = []
  for i in range(len(XYlist)-1):
    Ydis = XYlist[i+1][1] - XYlist[0][1]
    Xdis = XYlist[i+1][0] - XYlist[0][0]
    arc = np.arctan2(Ydis,Xdis)
    dis = Distance(Xdis,Ydis)
    arclist.append(arc)
    dislist.append(dis)

  print(arclist)
  print(dislist)

  data = pd.DataFrame()
  data["xy"] = XYlist[1:]
  data["arc"] = arclist
  data["dis"] = dislist

  print(data)

  data = data.sort_values(by=["arc","dis"],ascending=[True,True])

  res = list(data["xy"].values)
  res.insert(0, XYlist[0])
  print(res)
  return res

def CHall(img):
  _, bin = cv.threshold(img, 0.5, 255, cv.THRESH_BINARY)
  # 查找轮廓

  con,_ = cv.findContours(bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  # print(con)
  # print(len(con))
  if len(con) != 0:
    coor = []
    for i in range(len(con)):
      # print(len(con[i]))
      for item in con[i]:
        # print(item[0])
        coor.append([item[0][0],item[0][1]])

    # print(coor)
    print(len(coor))
    coor = np.array(coor)
    hull = cv.convexHull(coor)

    cv.fillPoly(img, [hull], (255, 255, 255))

    # 将图像转换为单通道灰度图像
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    return img
  else:
    return -999

