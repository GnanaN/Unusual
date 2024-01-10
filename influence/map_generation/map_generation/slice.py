import json
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
# from arcpy.sa import *
from PIL import Image
import numpy as np
import shutil
Image.MAX_IMAGE_PIXELS = None

#  第一个参数path是文件夹名称，第二个参数是放原始tif文件的文件夹名称
'''文件夹的组织方式如下，首先最外层文件夹，其路径为path，
   其中有一个文件夹放有原始tif文件，该文件夹的名称为pathname'''
def colorTif(tifpath,temp):
    # 新建临时文件夹和结果文件夹，如果他们存在就删掉重建，不存在新建
    g_ifltemp = temp + '\\g_ifltemp\\'
    if os.path.exists(g_ifltemp):
        shutil.rmtree(g_ifltemp)
    if not(os.path.exists(g_ifltemp)):
        os.makedirs(g_ifltemp)
    # 找到并读取原始图像

    img_path = tifpath
    # 转为整型后数据的保存路径
    Int_img_path =g_ifltemp +"RC_org.tif"
    # 如果转整型的结果不存在，再进行转整
    if not(os.path.exists(Int_img_path)):
        # print(Int_img_path)
        #转为整型的操作
        Int(img_path).save(Int_img_path)
    img = cv2.imread(Int_img_path,cv2.IMREAD_GRAYSCALE)
    #另外一种读取超大像素图的方法
    ''' mat = io.imread(img_path)
        print(mat.shape)'''

    # 灰度图转彩色图
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    B = img_color[:,:,0]
    G = img_color[:,:,1]
    R = img_color[:,:,2]
    print('-----------')
    print(B.shape)

    # 背景值设置为白色,所以RGB都为255
    B[B >= 4] = 255
    G[G >= 4] = 255
    R[R >= 4] = 255
    # B波段都为0
    B[B<4] = 0
    G[G == 0] = 168
    R[R == 0] = 56
    G[G == 1] = 209
    R[R == 1] = 139
    G[G == 2] = 128
    R[R == 2] = 255
    G[G == 3] = 0
    R[R == 3] = 255
    img = img_color
    img[:, :, 0] = B
    img[:, :, 1] = G
    img[:, :, 2] = R
    img = np.array(img)
    print(img_color.shape)
    print(img.shape)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    color_img_name = g_ifltemp +"RC_color.png"
    if  not(os.path.exists(color_img_name)):
        cv2.imwrite(color_img_name,img)
    # 返回彩色图片的路径
    return [color_img_name,tifpath]

def slice1(cresult,temp):
    # colortif_path = colorTif(path,pathname)
    trainpath =  temp+ '\\g_ifltemp\\train\\'
    if os.path.exists(trainpath):
        shutil.rmtree(trainpath)
    if not(os.path.exists(trainpath)):
        os.makedirs(trainpath)
    org_img = cv2.imread(cresult[0])
    height, width = org_img.shape[:2]
    print(height,width)
    prop = width/height
    print(prop)
    row, column = 15, round(15 * prop)
    print('height %d widht %d' % (row, column))

    row_step =  (int)(height/row)
    column_step = (int)(width/column)

    # print('row step %d col step %d'% (row_step, column_step))
    # print('height %d widht %d' % (row_step*row, column_step*column))

    img = org_img[0:height, 0:width]

    print(trainpath)
    if not (os.path.exists(trainpath)):
        os.makedirs(trainpath)

    for i in range(row):
        for j in range(column):
            pic_name = trainpath + '\\' + str(i) + "_" + str(j) + ".png"
            # print(pic_name)
            if (i == row-1 and j < column-1):
                tmp_img = img[(i * row_step):height, (j * column_step):(j * column_step) + column_step]
            elif (j == column-1 and i < row -1):
                tmp_img = img[(i * row_step):(i * row_step + row_step),(j * column_step):width]
            elif (i == row-1 and j == column-1):
                tmp_img = img[(i * row_step):height,(j * column_step):width]
            else:
                tmp_img = img[(i * row_step):(i * row_step + row_step),(j * column_step):(j * column_step) + column_step]
            cv2.imwrite(pic_name, tmp_img)
    #返回值为训练数据集的
    return [trainpath,cresult[1],row,column]

# 调用方式：
# if __name__ == "__main__":
#     org_tif = r"F:\CODE\influence\data\disaster"
#     # slice(path,org_tif)   #返回的是存放待训练数据的路径
#     slice1(colorTif(org_tif))

