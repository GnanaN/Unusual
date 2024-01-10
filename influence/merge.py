from osgeo import gdal, osr
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)
import shutil
import skimage.io as io

#读入数据
def mergeic(file_pathname,row,column,temp,org_tif):
    g_ifltemp = temp + '\\g_ifltemp'
    org_img = io.imread(org_tif)
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
            print('综合后', file_pathname + '\\' + filename)
            img = io.imread(file_pathname + '\\' + filename)
            if np.array_equal(np.unique(img), [118, 122, 137]):
                img[:] = [255, 255, 255]
            mask = np.all((img >= [200, 200, 200]) & (
                img <= [255, 255, 255]), axis=-1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img[mask, 3] = 0
            v.append(img)
        img = cv2.hconcat(v)
        h.append(img)
    img = cv2.vconcat(h)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA)
    resized_img = cv2.resize(img, (width, heigth), interpolation=cv2.INTER_LINEAR)
    mergepath = g_ifltemp + "\\" + 'merge.png'
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
    out_ds = driver.CreateCopy(savepath, ds, options=['COMPRESS=LZW'])
    #关闭文件
    ds = None
    out_ds = None
    return savepath

#
# if __name__ == '__main__':
#     path = r"F:\CODE\strength\data\xJ\inf\resa.tif"
#     m = mergeic(r"F:\0612\train",15,24,r"F:\0612\temp",path)
#     print(m)
#     addCoor(m,path,r"F:\0612\result")

