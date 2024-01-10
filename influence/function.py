from scipy import stats
import cv2
import numpy as np
from matplotlib import pyplot as plt
from rasterio.features import shapes
from sklearn.cluster import KMeans
plt.rcParams['font.sans-serif'] = ['SimHei']
from osgeo import gdal, ogr, osr
import os
from tifffile import imread
import geopandas as gpd
import jenkspy


def floderPrepare( ResultfloderPath, TempfloderPath):
    # 创建结果文件夹
    if not os.path.exists(ResultfloderPath):
        os.makedirs(ResultfloderPath)
        print("创建结果文件夹成功！")
    else:
        print("结果文件夹已存在！")
    # 创建临时文件夹
    if not os.path.exists(TempfloderPath):
        os.makedirs(TempfloderPath)
        print("创建临时文件夹成功！")
    else:
        print("临时文件夹已存在！")
    # 创建数据文件夹

def FindData(dataFloderPath):
    tifpath = []
    for root, dirs, files in os.walk(dataFloderPath):
        # 判断文件的后缀是否为.tif
        for file in files:
            if (file.split('.')[-1] == 'tif'):
                tifpath.append(os.path.join(root, file))
    return tifpath

def readImg(path):
    # 图像位深度是4 opencv无法读取，需要转换为16位深度
    img = imread(path)
    # img = np.array(img, dtype=np.float32)
    # img = cv2.GaussianBlur(img, (3, 3), 0)
    m,n = img.shape[0],img.shape[1]
    # 读取图片坐标系统
    dataset = gdal.Open(path)
    geotrans = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    return img, m, n, geotrans, proj

def find_outline(img):
    outline_img = np.zeros_like(img)
    np.array(outline_img, dtype=np.uint8)
    outline_img[img > 0] = 1
    return outline_img
def ChangeBackgValue(img):
    img[img < -100] = 9999
    img_min = np.nanmin(img)
    img[img == 9999] = img_min
    img[img > 9999] = img_min
    img[np.isnan(img)] = img_min
    # print('背景值修改成功')

    return img

# 对八位的不需要进行综合的数据进行预处理

def ChangeBackgValue_noge(img):
    img[img < 1] = 0
    img[img == 255] = 0
    return img
# 高斯滤波


def GaussBlur(img, ratio=0.01):
    '''
    高斯滤波降噪
    高斯滤波器的长和宽必须是正奇数，如3、5、7等
    返回值：高斯滤波后的图像
    '''
    m, n = img.shape[0], img.shape[1]
    ksize = int(min(m, n)*ratio) if int(min(m, n) *
                                        ratio) % 2 != 0 else int(min(m, n)*ratio)+1
    # print('高斯滤波器大小为：', ksize)
    return cv2.GaussianBlur(img, (ksize, ksize), 0), ksize

# 对图像进行kmeans聚类并展示结果


def kCluster(img, cluster_num):
    '''
    img:输入的图像
    cluster_num:聚类数目
    返回值:聚类后的labels
    '''
    # kmeans聚类
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(
        img.reshape(-1, 1))
    labels = kmeans.labels_
    return labels

# 聚类标签转为ndarray矩阵


def TransToArray(img, m, n, clusterlabel):  # 默认最大值所对应的类，应该是灾害所在的区域
    '''
    聚类标签转为矩阵
    :param m: 原始数据行数
    :param n: 原始数据列数
    :param clusterlabel: 聚类结果标签
    :param img_data: 转换为1列的原始数据
    :return: temparray: 矩阵转换结果
    '''
    img_data = img.reshape((m * n, 1))
    temparray = np.zeros((m, n, 1))
    count = 0
    maxLoc = np.argmax(img_data)
    # print(maxLoc)
    # print(clusterlabel.shape)
    # 判断最大值所在区域的类型
    cluster_val = clusterlabel[maxLoc]
    label0 = np.array(0)
    label1 = np.array(1)

    for i in range(m):
        for j in range(n):
            if (clusterlabel[count] == cluster_val):
                temparray[i][j] = label1
            else:
                temparray[i][j] = label0
            count += 1

    # print(temparray)
    return temparray


def reclass(img, classNum,method='equal_interval'):
    '''
    img:输入的图像
    classNum:分类数目
    返回值:重分类后的图像
    '''
    max_val = np.max(img)
    classified_img = np.zeros_like(img, dtype=np.uint8)
    # 等间隔分级
    if method == 'equal_interval':
        # 分类和赋值
        # 这里示例将图像分成五个区间进行分类，可以根据具体情况调整分级
        breaks = np.linspace(0, 1, classNum+1)  # 根据需要分成五类，这里是0-1之间的五个分界点
    else:  # 自然断点法

        data = np.array(img[img>0])
        flattened_data = data.flatten()
        # 重采样数据
        if len(flattened_data)  > 25000000:
            sample_size = int(len(flattened_data) * 0.025)  # 使用原始数据的5%
            subset = np.random.choice(
            flattened_data, size=sample_size, replace=False)
        else:
            sample_size = int(len(flattened_data) * 0.05)  # 使用原始数据的5%
            subset = np.random.choice(flattened_data, size=sample_size, replace=False)
        breaks = jenkspy.jenks_breaks(subset, 5)
        breaks[5] = (max_val)
    print(breaks)
    for i in range(classNum):
        classified_img[(img >= breaks[i]) & (img < breaks[i+1])] = i+1
    # 将classified_img转为int-8类型
    # classified_img = classified_img.astype(np.uint8)
    return classified_img



# 计算外接多边形,确定多边形方向


def calDirection( input_img):
    # 假设img是一个二值图像，其中斑块的像素值为1，其余像素值为0
    # 找到所有斑块的像素
    # plt.imshow(org_img, cmap='viridis')
    # plt.title("grade" + str(level))
    points = np.argwhere(input_img == 1)
    # 可视化斑块位置
    # plt.imshow(input_img, cmap='gray')
    # plt.plot(points[:, 1], points[:, 0], 'b.', markersize=1)

    # 计算最小外接矩形
    rect = cv2.minAreaRect(points)

    # 获取矩形的四个顶点，并将其转换为整数
    box = cv2.boxPoints(rect)
    # box = np.int0(box)

    # 计算斑块的形状（宽度和高度的比例）
    shape = rect[1][0] / \
        rect[1][1] if rect[1][0] < rect[1][1] else rect[1][1] / rect[1][0]

    # 计算斑块的方向（矩形的旋转角度）
    width, height = rect[1]
    if width > height:
        direction = rect[2] - 90
        width, height = height, width
    else:
        direction = rect[2]
    shape = width / height

    # 可视化最小外接矩形
    # plt.plot(box[:, 1], box[:, 0], color='red', linewidth=2)

    # 可视化方向direction
    # plt.plot([rect[0][1], rect[0][1] + np.cos(np.deg2rad(direction)) * 50],
    #          [rect[0][0], rect[0][0] + np.sin(np.deg2rad(direction)) * 50], color='red', linewidth=2)
    # # 显示图像
    # plt.savefig("C:\\Users\\nana_\\Desktop\\组会20231130\\" + str(level)+".png")
    # plt.show()
    # print(direction)
    return direction

# 右斜核


def incline_kernel(arr, cellsize):
    incline_width = cellsize // 2 if cellsize % 2 == 0 else cellsize // 2 + 1
    for m in range(incline_width):
        for n in range(incline_width-m-1):
            arr[m][n] = 0
    for m in range(incline_width, cellsize):
        for n in range(cellsize + incline_width - m - 1, cellsize):
            arr[m][n] = 0
    return arr
# 一下三个是自定义核的三个函数
# 水平核


def horizontal_kernel(arr, cellsize):
    width = cellsize // 2 if cellsize % 2 == 0 else cellsize // 2 + 1
    for m in range(cellsize-width+1, cellsize):
        for n in range(cellsize):
            arr[m][n] = 0
    return arr

# 竖直核


def vertical_kernel(arr, cellsize):
    width = cellsize // 2 if cellsize % 2 == 0 else cellsize // 2 + 1
    for m in range(cellsize):
        for n in range(cellsize-width+1, cellsize):
            arr[m][n] = 0
    return arr

# 根据角度自定义核形状


def designKernel(direction, cellsize):
    # 设置一个cellsize*cellsize的核，值为1
    kernel = np.ones((cellsize, cellsize), np.uint8)
    if direction >= -67.5 and direction < -22.5:  # 右边斜
        # print('顺时针核')
        closed_kernel = np.rot90(incline_kernel(kernel, cellsize), k=1)
    if direction > 22.5 and direction <= 67.5:  # 左边斜
        # print('逆时针核')
        closed_kernel = incline_kernel(kernel, cellsize)
    if direction >= -22.5 and direction <= 22.5:  # 横着的
        # print('平行核')
        closed_kernel = horizontal_kernel(kernel, cellsize)
    if (direction > 67.5 and direction <= 90) or (direction >= -90 and direction < -67.5):  # 竖着的
        # print('垂直核')
        closed_kernel = vertical_kernel(kernel, cellsize)
    return closed_kernel

# 开闭运算 用于分等级综合
def OpenClose(img, d_kernel):
    """
    开闭运算
    :param img: 图像路径
    :param kernel_cell: 开闭运算核大小
    :return: closed: 开闭运算结果
    """
    closed = img
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, d_kernel)
    # closed = np.array(closed, np.uint8)
    # closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, d_kernel)
    # closed = cv2.erode(closed, None, iterations=5)
    # closed = cv2.dilate(closed, None, iterations=5)
    return closed

# 用轮廓提取，删除一些细小的斑块
def OpenClose1(img, d_kernel):
    """
    开闭运算
    :param img: 图像路径
    :param kernel_cell: 开闭运算核大小
    :return: closed: 开闭运算结果
    """
    closed = img
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, d_kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, d_kernel)
    # closed = cv2.erode(closed, None, iterations=5)
    # closed = cv2.dilate(closed, None, iterations=5)

    return closed

# 分层处理数据
def LayerProcess(img,mask, cellsize, classNum=5,):
    resultimg = np.zeros_like(img)
    for level in range(classNum, 0, -1):
        print(level)
        if (level in img):
            img_copy = img.copy()
            img_copy[img_copy < level] = 0
            img_copy[img_copy >= level] = 1
            direction = calDirection(img_copy)
            img_closed_bydesignkernel = OpenClose(
                img_copy, designKernel(direction, cellsize))
            img_closed_bydesignkernel = img_closed_bydesignkernel*mask

            img_closed_bydesignkernel = select_plaque(
                img_closed_bydesignkernel,level)
            resultimg = cv2.addWeighted(
                resultimg, 1, img_closed_bydesignkernel, 1, 0)
        else:
            img_closed_bydesignkernel = np.zeros_like(img)
            img_closed_bydesignkernel[resultimg>0] = 1
            resultimg = cv2.addWeighted(
                resultimg, 1, img_closed_bydesignkernel, 1, 0)
            print("%s is not in img" % level)

    return resultimg


def LayerProcess2_noremove(img, mask, cellsize, classNum=5):
    resultimg = np.zeros_like(img)
    for level in range(classNum, 0, -1):
        # print(level)
        if (level in img):
            # print('ok')
            img_copy = img.copy()
            img_copy[img_copy < level] = 0
            img_copy[img_copy >= level] = 1
            direction = calDirection(img_copy)
            img_closed_bydesignkernel = OpenClose(
                img_copy, designKernel(direction, cellsize))
            img_closed_bydesignkernel *= mask
            resultimg = cv2.addWeighted(
                resultimg, 1, img_closed_bydesignkernel, 1, 0)
        else:
            img_closed_bydesignkernel = np.zeros_like(img)
            img_closed_bydesignkernel[resultimg > 0] = 1
            resultimg = cv2.addWeighted(
                resultimg, 1, img_closed_bydesignkernel, 1, 0)
            print("%s is not in img" % level)
    return resultimg

# 移除面积小于阈值的斑块
def remove_small_area1(img_, area_name):
    # 读取图像
    img = img_
    max = int(np.array(img).max())

    for i in range(1, max+1):  # 遍历每个等级，最高等级除外
        labels = []
        if i < max:
            area1 = area_name
        else:
            area1 = area_name//3
        img_copy = img.copy()
        img_copy[img_copy != i] = 0
        img_copy[img_copy == i] = 255
        img_copy = cv2.convertScaleAbs(img_copy)
        num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(
            img_copy, connectivity=4, ltype=cv2.CV_32S)
        # print('斑块数量为：', num_labels)
        for t in range(1, num_labels, 1):
            x, y, w, h, area = stats[t]
            if area < area1:
                index = np.where(labels == t)
                labels[index[0], index[1]] = -i
            else:
                index = np.where(labels == t)
                labels[index[0], index[1]] = 0
        img = img[:] + labels[:]
        img = np.array(img, np.uint8)
    return img

def TiftoShp(path, field_name, outpath):
    inraster = gdal.Open(path)  # 读取路径中的栅格数据
    inband = inraster.GetRasterBand(1)  # 这个波段就是最后想要转为矢量的波段，如果是单波段数据的话那就都是1
    prj = osr.SpatialReference()
    prj.ImportFromWkt(inraster.GetProjection())  # 读取栅格数据的投影信息，用来为后面生成的矢量做准备

    outshp = outpath  # 给后面生成的矢量准备一个输出文件名，这里就是把原栅格的文件名后缀名改成shp了
    drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outshp):  # 若文件已经存在，则删除它继续重新做一遍
        drv.DeleteDataSource(outshp)
    Polygon = drv.CreateDataSource(outshp)  # 创建一个目标文件
    Poly_layer = Polygon.CreateLayer(
        outshp, srs=prj, geom_type=ogr.wkbMultiPolygon)  # 对shp文件创建一个图层，定义为多个面类
    # 给目标shp文件添加一个字段，用来存储原始栅格的pixel value,浮点型，
    newField = ogr.FieldDefn(field_name, ogr.OFTInteger)
    Poly_layer.CreateField(newField)

    gdal.Polygonize(inband, None, Poly_layer, 0)  # 核心函数，执行的就是栅格转矢量操作

    # 删除ignore_value链表中的类别要素
    for feature in Poly_layer:
        class_value = feature.GetField(field_name)
        if (class_value == 0):
            # 通过FID删除要素
            Poly_layer.DeleteFeature(feature.GetFID())
        if (class_value == -1):
            # 通过FID删除要素
            Poly_layer.DeleteFeature(feature.GetFID())

    # gdal.FPolygonize(inband, None, Poly_layer, 0)  # 只转矩形，不合并
    Polygon.SyncToDisk()
    Polygon = None
    print("创建文件成功！")
    return outshp

def ReadShp(ShpPath):
    # 支持中文路径
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    # 支持中文编码
    gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")
    # 注册所有的驱动
    ogr.RegisterAll()
    # 打开数据
    ds = ogr.Open(ShpPath, 0)
    if ds == None:
        return ("打开文件失败！")
    # 获取数据源中的图层个数，shp数据图层只有一个，gdb、dxf会有多个
    # iLayerCount = ds.GetLayerCount()
    # 获取第一个图层
    oLayer = ds.GetLayerByIndex(0)
    if oLayer == None:
        return ("获取图层失败！")

    extent = oLayer.GetExtent()
    # print('extent:', extent)
    # print('ul:', extent[0], extent[3])
    # print('lr:', extent[1], extent[2])

    ds.Destroy()
    del ds
    # print('数据范围为：', extent)
    return extent

def calConvexHull(input_img):
    points = np.argwhere((input_img > 0) & (input_img < 4))
    # points = np.argwhere(input_img > 0)
    hull = cv2.convexHull(points)
    hull = hull[:, :, ::-1]
    # 创建一个与 input_img 大小相同的空白图像
    img = np.zeros_like(input_img)
    # 不一定要填充为1，可以填充成等级
    # 统计input_img中除了0以外的众数

    # 将多边形区域填充为1
    cv2.fillPoly(img, [hull],1)
    return img

def WriteTifImg(SavePath, im_proj, im_geotrans, img, datatype=None):
    '''功能：用于写TIF格式的遥感图像，同时兼容一个通道 和 三个通道
       返回值：im_proj : 地图投影信息，保持与输入图像相同
             im_geotrans : 仿射矩阵,计算当前图像块的仿射信息
             im_data：通道顺序位 [channel,height,width]， 当前图像块的像素矩阵，
             datatype：指定当前图像数据的数据类型，默认和输入的im_data类型相同'''
    # gdal数据类型包括
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64
    # 判断栅格数据的数据类型
    if img.dtype == np.uint8:
        datatype = gdal.GDT_Byte
    elif img.dtype == np.int16:
        datatype = gdal.GDT_Int16
    elif img.dtype == np.uint16:
        datatype = gdal.GDT_UInt16
    elif img.dtype == np.uint32:
        datatype = gdal.GDT_UInt32
    elif img.dtype == np.int32:
        datatype = gdal.GDT_Int32
    elif img.dtype == np.float32:
        datatype = gdal.GDT_Float32
    elif img.dtype == np.float64:
        datatype = gdal.GDT_Float64
    else:
        raise ValueError(f"Unsupported data type: {img.dtype}")
    # 判读数组维数
    if len(img.shape) == 3:
        im_height, im_width, im_bands = img.shape
    else:
        im_bands, (im_height, im_width) = 1, img.shape

    # print("波段数为", im_bands)
    # print("长宽为",im_height, im_width)

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(SavePath, im_width, im_height, im_bands, datatype)
    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(img)  # 写入数组数据
    else:
        for i in range(im_bands):  # 按波段写入
            dataset.GetRasterBand(i + 1).WriteArray(img[i])
    del dataset


def imgbymask(img, mask=None):
    min_val = np.min(img)
    max_val = np.max(img)
    if min_val == -99:
        min_val = 0
    else:
        min_val = min_val
    '''
    print('最小值，最大值',min_val, max_val)
    normalized_img = (img - min_val) / (max_val - min_val)  # 归一化处理
    if mask is not None:
        print('有mask')
        normalized_img[mask == 0] = -1
    '''
    normalized_img = img
    return normalized_img

def select_plaque(img,level):
    labels=[]
    if level <= 3:
        num = 3+2
    else:
        num = level+2
    # print(num)
    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(
        img, connectivity=4, ltype=cv2.CV_32S)
    # 筛选选面积前三的斑块
    area = stats[:, 4]
    # print(area)
    labellist = np.argsort(area)[::-1]
    if len(labellist) > num-1:
        for t in labellist[:num]:
            index = np.where(labels == t)
            labels[index] = 0
        for t in labellist[num:]:
            index = np.where(labels == t)
            labels[index] = -1
    else:
        for t in labellist:
            index = np.where(labels == t)
            labels[index] = 0
    img = img[:] + labels[:]
    img = np.array(img, np.uint8)
    return img

def delete_shp(shppath):
    # 读取shp文件
    data = gpd.read_file(shppath, encoding='utf-8')
    # 计算data中的面积
    data['area'] = data.area
    # 按属性选择value=0的要素
    index = []
    for i in range(1, 6):
        try:
            mask = data['Value'] == i
            # 找到value值为i的记录
            index.extend(data[mask].sort_values(by='area', ascending=False).index[10:])
        except:
            print('没有第%s个等级的斑块' % i)
    # print(len(index))
    data.drop(index=index, inplace=True)
    # 保存删除后的shp文件
    data.to_file(shppath, encoding='utf-8')

def edit_value(shppath,grade):
    # 读取shp文件，修改value字段为grade
    data = gpd.read_file(shppath, encoding='utf-8')
    data['Value'] = grade
    # 保存修改后的shp文件
    data.to_file(shppath, encoding='utf-8')


def LayerProcessScale2Group3(img, mask, cellsize, classNum=5,):
    resultimg = np.zeros_like(img)
    for level in range(classNum, 0, -1):
        print(level)
        # 遍历每个等级，如果存在这个等级就综合，不存在就将现有斑块的值都加1
        if (level in img):
            img_copy = img.copy()
            img_copy[img_copy < level] = 0
            img_copy[img_copy >= level] = 1
            # 计算核方向
            # direction = calDirection(img_copy)
            # 设计核，并进行开闭运算，
            img_closed_bydesignkernel = OpenClose(
                img_copy, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cellsize, cellsize)))
            # 将运算结果与mask相乘，删除多余区域
            # img_closed_bydesignkernel = img_closed_bydesignkernel*mask
            img_closed_bydesignkernel = np.array(
                img_closed_bydesignkernel, np.uint8)
            # 叠加起来
            img_closed_bydesignkernel = remove_small_area2(img_closed_bydesignkernel,400)
            resultimg = cv2.addWeighted(
                resultimg, 1, img_closed_bydesignkernel, 1, 0)
        else:
            img_closed_bydesignkernel = np.zeros_like(img)
            img_closed_bydesignkernel[resultimg > 0] = 1
            resultimg = cv2.addWeighted(
                resultimg, 1, img_closed_bydesignkernel, 1, 0)
            print("%s is not in img" % level)
    return resultimg

# 课题组三专用，删除小面积斑块
def remove_small_area2(img_, area_size):
    # 读取图像
    img_copy = img_.copy()
    img_copy[img_copy != 1] = 0
    img_copy = cv2.convertScaleAbs(img_copy)
    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(
    img_copy, connectivity=4, ltype=cv2.CV_32S)
    for t in range(1, num_labels, 1):
        x, y, w, h, area = stats[t]
        if area < area_size:
            index = np.where(labels == t)
            labels[index[0], index[1]] = -1
        else:
            index = np.where(labels == t)
            labels[index[0], index[1]] = 0
    img = img_copy[:] + labels[:]
    img = np.array(img, np.uint8)
    return img



def hullOfgrade(path):
    '''
    tif:综合后的tif路径
    gradelsit:需要提取凸包的等级列表
    '''

