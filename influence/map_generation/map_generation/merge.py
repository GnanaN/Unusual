from osgeo import gdal, osr
import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
import numpy as np
import shutil


#读入数据
def mergeic(file_pathname,row,column,temp,org_tif):
    g_ifltemp = temp + '\\g_ifltemp'
    org_img = cv2.imread(org_tif)
    heigth = org_img.shape[0]
    width = org_img.shape[1]

    # if (os.path.exists(g_ifltemp)):
    #     shutil.rmtree(g_ifltemp)
    # if not (os.path.exists(g_ifltemp)):
    #     os.makedirs(g_ifltemp)

    h = []
    for r in range(row):
        v = []
        for c in range(column):
            # ！！！应该是fake_B
            filename = str(r) + '_' + str(c) + '_fake_B' + '.png'
            # filename = str(r) + '_' + str(c) + '.png'
            # print(file_pathname + '\\' + filename)
            img = cv2.imread(file_pathname + '\\' + filename)
            v.append(img)
        img = cv2.hconcat(v)
        h.append(img)
    img = cv2.vconcat(h)
    resized_img = cv2.resize(img, (width,heigth), interpolation=cv2.INTER_LINEAR)
    B = resized_img[:,:,0]
    G = resized_img[:,:,1]
    R = resized_img[:,:,2]
    B[ B == 122] = 255
    G[ G == 137] = 255
    R[ R == 118] = 255
    img = resized_img
    img[:, :, 0] = B
    img[:, :, 1] = G
    img[:, :, 2] = R
    img = np.array(img)
    # print(heigth,width)
    mergepath = g_ifltemp+ "\\"+'merge.tif'
    cv2.imwrite(mergepath,resized_img)
    print(mergepath)
    return mergepath

def addCoor(mergepath,org_path,result):
    g_iflres = result + '\\g_iflres\\'
    if os.path.exists(g_iflres):
        shutil.rmtree(g_iflres)
    if not (os.path.exists(g_iflres)):
        os.makedirs(g_iflres)
    savepath = g_iflres + 'result.tif'
    #打开tif文件
    ds = gdal.Open(mergepath)
    print(ds)
    #获取文件的投影信息
    #创速一个新的坐标系对象
    srs = osr.SpatialReference()
    inraster = gdal.Open(org_path)   # 读取路径中的栅格数据
    gt = inraster.GetGeoTransform()
    srs.ImportFromWkt(inraster.GetProjection())
    # srs.ImportFromEPSG(3857)#这里使用wGS84坐标系作为示钢
    #更新地理交换信息
    ds.SetGeoTransform(gt)
    #更新投影信息
    ds.SetProjection(srs.ExportToWkt())
    #保存修改后的文件
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.CreateCopy(savepath, ds)
    #关闭文件
    ds = None
    out_ds = None
    return savepath
