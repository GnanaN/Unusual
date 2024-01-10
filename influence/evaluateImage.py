'''
基于综合前后的图像(tif, png)进行综合质量评价
吕开来
20231211
'''
import struct
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import numpy as np
import cv2
import time
import pandas as pd


#2023年底示例采用的colorList，颜色为BGR格式
Color_List_Strength_3 = [[0,168,56],[0,255,255],[0,0,255]]
Color_List_Influence_4 = [[0,168,56],[0,209,139],[0,128,255],[0,0,255]]
Color_List_Strength_5 = [[0,168,56],[0,209,139],[0,255,255],[0,128,255],[0,0,255]]
Color_List_Strength_10 = [[0,97,0],[0,128,60],[0,161,107],[0,196,164],[0,235,223],[0,234,255],[0,187,255],[0,145,255],[0,98,255],[0,34,255]]


def evaluateLvTif(path0: str, path1: str, tempPath: str) -> list:
    '''
    对综合前后的等级tif进行综合质量评价, 并输出评价指标数组

    Parameters
    ----------
    path0 : str
        综合前等级tif栅格的路径
    img1 : str
        综合后等级tif栅格的路径
    tempPath : str
        用于保存评价png的临时目录

    Returns
    -------
    evaOutList : list
        评价指标数组, 内容依次为边缘密度、边缘密度改善率、位置准确度、压缩率、压缩率改善率
    '''


    # stTime = time.time()
    # 读取数据
    img0 = cv2.imread(path0,-1)
    # img1 = cv2.imread(path1,-1)
    lvList = np.unique(img0)
    breakNum = int(lvList[-1])
    # 确定色彩映射表
    if (breakNum <= 5):
        colorList = Color_List_Strength_5
    elif (breakNum <= 10):
        colorList = Color_List_Strength_10
    else:
        print('错误! 输入evaluateLvTif()的tif等级大于10, 可能不是等级tif.')
        return None

    # 等级tif转为彩色png
    pngPath0 = tempPath + '\\' + path0.split('\\')[-1].split('.')[0] + '.png'
    pngPath1 = tempPath + '\\' + path1.split('\\')[-1].split('.')[0] + '.png'
    lvTif2ColorPng(path0, pngPath0, breakNum, colorList)
    lvTif2ColorPng(path1, pngPath1, breakNum, colorList)

    # 进行评价, 内容依次为边缘密度、边缘密度改善率、位置准确度、压缩率、压缩率改善率
    ed0 = calPngED(pngPath0)
    ed1 = calPngED(pngPath1)
    cr0 = calPngCR(pngPath0)
    cr1 = calPngCR(pngPath1)
    pa = calPngPA(pngPath0, pngPath1, colorList)
    evaOutList = []
    evaOutList.append(ed1)
    evaOutList.append((ed0-ed1)/ed0)
    evaOutList.append(pa)
    evaOutList.append(cr1)
    evaOutList.append((cr0-cr1)/cr0)
    # print('评价一组tif用时为', time.time()-stTime)

    return evaOutList

def lvTif2ColorPng(path: str, outPath: str, break_num: int, colorList: list) -> None:
    '''
    将等级tif转为彩色png并保存

    Parameters
    ----------
    path : str
        等级tif栅格的路径
    outPath : str
        彩色png图片的路径
    break_num : int
        等级数
    colorList : list
        色彩映射表，各颜色应为[B,G,R]格式

    Returns
    -------
    None
    '''

    #打开原始灰度tif栅格
    imgRaw = cv2.imread(path,-1)
    breakList = []
    for i in range(break_num+1):
        breakList.append(i)

    #基于分类断点值列表进行重分类和色彩渲染
    imgBGR = np.zeros((imgRaw.shape[0], imgRaw.shape[1], 3), dtype="uint8")
    for i in range(len(breakList)-1):
        imgBGR[(imgRaw[:,:] > breakList[i]) & (imgRaw[:,:] <= breakList[i+1])] = colorList[i]
    # #查看过程图像
    # cv2.imshow('title1',imgBGR)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite(outPath, imgBGR)

def grayTif2ColorPng(path: str, outPath: str, breakList: list, colorList: list) -> None:
    '''
    将连续灰度tif转为彩色png并保存

    Parameters
    ----------
    path : str
        等级tif栅格的路径
    outPath : str
        彩色png图片的路径
    breakList : list
        分类断点值列表，其中应包括最大值和最小值
    colorList : list
        色彩映射表，各颜色应为[B,G,R]格式

    Returns
    -------
    None
    '''

    #打开原始灰度tif栅格
    imgRaw = cv2.imread(path,-1)

    #基于分类断点值列表进行重分类和色彩渲染
    breakList[0] = breakList[0] - 0.0001
    imgBGR = np.zeros((imgRaw.shape[0], imgRaw.shape[1], 3), dtype="uint8")
    for i in range(len(breakList)-1):
        imgBGR[(imgRaw[:,:] > breakList[i]) & (imgRaw[:,:] <= breakList[i+1])] = colorList[i]
    # #查看过程图像
    # cv2.imshow('title1',imgBGR)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite(outPath, imgBGR)

def calPngED(path: str) -> float:
    '''
    计算单张png图片的边缘密度
    '''
    img = cv2.imread(path)
    pixelSum = img.size
    x, y= img.shape[0:2]
    edgeDensity = -1

    #计算边缘密度edgeDensity
    edge = cv2.Canny(img,0,5)  #指定canny算子上下阈值

    # #查看边缘图像
    # cv2.imshow('calLeg',edge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    edgeSum = 0
    for i in range(x):
        for j in range(y):
            if edge[i,j]==255:
                edgeSum += 1
    edgeDensity = edgeSum/pixelSum

    # #查看过程图像
    # # cv2.imshow('title0',imgRGB)
    # cv2.imshow('title1',edge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return edgeDensity

def calPngCR(path: str) -> float:
    '''
    计算单张png图片的压缩率
    '''
    Raster_Width = 0  #图像宽度
    Raster_Length = 0  #图像长度
    Bits_Per_Sample = 0  #像素位数
    Color_Space = -1  #色彩空间类型
    Color_Space_Count_Dict = {0:1, 2:3, 4:2, 6:4}  #色彩空间类型对应的颜色通道数
    Pixel_Byte_Count = 0  #图像部分大小
    Estimated_Pixel_Byte_Count = 0  #图像部分估计大小

    Input_File_Name = path
    fo = open(Input_File_Name,'rb')

    Header_start_Offset = 0  #IHDR标记后的偏移量
    Pixel_Byte_start_Offset = []  #IDAT标记后的偏移量

    #遍历获取标记的位置
    i = fo.read(1)
    while (i):
        if (i == b'\x49'):
            i_next = fo.read(3)
            if (i_next == b'\x44\x41\x54'):
                Pixel_Byte_start_Offset.append(fo.tell())
            elif (i_next == b'\x48\x44\x52'):
                Header_start_Offset = fo.tell()
            else:
                fo.seek(-3,1)
        i = fo.read(1)

    fo.seek(Header_start_Offset,0)
    Raster_Width = struct.unpack('>L',fo.read(4))[0]
    Raster_Length = struct.unpack('>L',fo.read(4))[0]
    Bits_Per_Sample = struct.unpack('b',fo.read(1))[0]
    Color_Space = struct.unpack('b',fo.read(1))[0]

    #IDAT标记可能有多个
    for i in Pixel_Byte_start_Offset:
        fo.seek(i-8,0)
        Pixel_Byte_Count += struct.unpack('>L',fo.read(4))[0]  #累加各IDAT块的长度即为图像部分大小

    Estimated_Pixel_Byte_Count = Raster_Width * Raster_Length * (Bits_Per_Sample*Color_Space_Count_Dict[Color_Space]/8)
    Compression_Rate = Pixel_Byte_Count / Estimated_Pixel_Byte_Count
    fo.close()

    return Compression_Rate

def calTifCR(path: str) -> float:
    '''
    计算单张tif图片的压缩率
    '''
    Band_Count = 0  #图像数
    Raster_Width = []  #图像宽度
    Raster_Length = []  #图像长度
    Bits_Per_Sample = []  #像素位数
    Color_Space = []  #色彩空间类型
    Compression_Type = []  #压缩类型
    Tiff_Field_Type_Dict = {1:'b', 2:'c', 3:'h', 4:'l', 5:'RATIONAL'}
    Tiff_Field_Length_Dict = {1:1, 2:1, 3:2, 4:4}
    Pixel_Byte_Count = 0  #图像部分大小
    Estimated_Pixel_Byte_Count = 0  #图像部分估计大小
    IFD_Count = 0         #IFD的总数
    IFD_Offset_List = []  #各IFD的偏移量
    IFD_DE_List = []      #各IFD的DE内容

    #判断tif文件版本
    fo = open(path,'rb')
    fo.seek(2,0)
    Version = struct.unpack('h',fo.read(2))[0]

    #根据tif文件版本计算压缩率
    if (Version == 42):  #经典Tiff
        #遍历各IFD,读取DE
        fo.seek(4,0)
        IFD_Offset_Next = struct.unpack('l',fo.read(4))[0]  #获取第一个IFD的偏移量
        while (IFD_Offset_Next != 0):
            IFD_Offset_List.append(IFD_Offset_Next)
            IFD_Count += 1
            fo.seek(IFD_Offset_Next,0)
            IFD_DE_Count = struct.unpack('h',fo.read(2))[0]  #获取DE数量
            #读取DE
            aDict = {}
            for i in range(IFD_DE_Count):
                aTuple = struct.unpack('2H2l',fo.read(12))
                if (aTuple[2]==1 and aTuple[0]!=325 and aTuple[0]!=279):
                    aDict[aTuple[0]] = aTuple[3]
                else:
                    aDict[aTuple[0]] = aTuple[1:]
            IFD_DE_List.append(aDict)
            IFD_Offset_Next = struct.unpack('l',fo.read(4))[0]  #获取下一个IFD的偏移量
        # print(IFD_DE_List)

        #根据DE获取度量结果
        Band_Count = IFD_Count
        for i in range(IFD_Count):
            Raster_Width.append(IFD_DE_List[i][256])
            Raster_Length.append(IFD_DE_List[i][257])
            #如果像素位数为多个值,需根据指针查找该元组
            if (isinstance(IFD_DE_List[i][258],tuple)):
                fo.seek(IFD_DE_List[i][258][2],0)
                tupleLen = IFD_DE_List[i][258][1]
                aTuple = struct.unpack(str(tupleLen)+'h',fo.read(tupleLen*2))
                Bits_Per_Sample.append(aTuple)
            else:
                Bits_Per_Sample.append(IFD_DE_List[i][258])
            Color_Space.append(IFD_DE_List[i][262])
            Compression_Type.append(IFD_DE_List[i][259])
            #计算图像部分大小
            if (279 in IFD_DE_List[i]):
                #如果没有采用瓦片切片存储
                stripNum = IFD_DE_List[i][279][1]  #获取条带大小列表的长度
                if (stripNum == 1):
                    #条带大小可能存为单个值
                    Pixel_Byte_Count = IFD_DE_List[i][279][2]
                else:
                    stripByteOffset = IFD_DE_List[i][279][2]  #获取条带大小列表的偏移量
                    stripType = Tiff_Field_Type_Dict[IFD_DE_List[i][279][0]]  #获取条带值的存储类型
                    stripLength = Tiff_Field_Length_Dict[IFD_DE_List[i][279][0]]  #获取条带值的字节长度
                    fo.seek(stripByteOffset,0)
                    stripByteTuple = struct.unpack(str(stripNum)+stripType,fo.read(stripNum*stripLength))  #读取条带大小列表
                    for j in range(stripNum):
                        Pixel_Byte_Count += stripByteTuple[j]  #累加条带大小列表中的值即可得到图像部分大小
            elif (325 in IFD_DE_List[i]):
                #如果采用了瓦片切片存储
                tileNum = IFD_DE_List[i][325][1]  #获取瓦片大小列表的长度
                if (tileNum==1):
                    #瓦片大小可能存为单个值
                    Pixel_Byte_Count = IFD_DE_List[i][325][2]
                else:
                    tileByteOffset = IFD_DE_List[i][325][2]  #获取瓦片大小列表的偏移量
                    tileType = Tiff_Field_Type_Dict[IFD_DE_List[i][325][0]]  #获取瓦片值的存储类型
                    tileLength = Tiff_Field_Length_Dict[IFD_DE_List[i][325][0]]  #获取瓦片值的字节长度
                    fo.seek(tileByteOffset,0)
                    tileByteTuple = struct.unpack(str(tileNum)+tileType,fo.read(tileNum*tileLength))  #读取瓦片大小列表
                    #print(tileByteTuple)
                    for j in range(len(tileByteTuple)):
                        Pixel_Byte_Count += tileByteTuple[j]  #累加瓦片大小列表中的值即可得到图像部分大小
            else:
                print("存在未识别的tif存储结构, tif压缩率计算有误!")
                return -1.0

        #计算图像部分估计大小
        for i in range(Band_Count):
            if (type(Bits_Per_Sample[i])==tuple):
                Bits_Per_Sample_Sum = 0
                for j in Bits_Per_Sample[i]:
                    Bits_Per_Sample_Sum += j
            else:
                Bits_Per_Sample_Sum = Bits_Per_Sample[i]
            Estimated_Pixel_Byte_Count += Raster_Width[i] * Raster_Length[i] * (Bits_Per_Sample_Sum/8)

        Compression_Rate = Pixel_Byte_Count / Estimated_Pixel_Byte_Count
        fo.close()
        return Compression_Rate
    elif (Version == 43):  #BigTiff
        #遍历各IFD,读取DE
        fo.seek(8,0)
        IFD_Offset_Next = struct.unpack('q',fo.read(8))[0]  #获取第一个IFD的偏移量
        while (IFD_Offset_Next != 0):
            IFD_Offset_List.append(IFD_Offset_Next)
            IFD_Count += 1
            fo.seek(IFD_Offset_Next,0)
            IFD_DE_Count = struct.unpack('q',fo.read(8))[0]  #获取DE数量
            #读取DE
            aDict = {}
            for i in range(IFD_DE_Count):
                aTuple0 = struct.unpack('2H',fo.read(4))
                aTuple1 = struct.unpack('2q',fo.read(16))
                aTuple = aTuple0 + aTuple1
                if (aTuple[2]==1 and aTuple[0]!=325 and aTuple[0]!=279):
                    aDict[aTuple[0]] = aTuple[3]
                else:
                    aDict[aTuple[0]] = aTuple[1:]
            IFD_DE_List.append(aDict)
            IFD_Offset_Next = struct.unpack('l',fo.read(4))[0]  #获取下一个IFD的偏移量
        # print(IFD_DE_List)

        #根据DE获取度量结果
        Band_Count = IFD_Count
        for i in range(IFD_Count):
            Raster_Width.append(IFD_DE_List[i][256])
            Raster_Length.append(IFD_DE_List[i][257])
            #如果像素位数为多个值,需根据指针查找该元组
            if (isinstance(IFD_DE_List[i][258],tuple)):
                fo.seek(IFD_DE_List[i][258][2],0)
                tupleLen = IFD_DE_List[i][258][1]
                aTuple = struct.unpack(str(tupleLen)+'h',fo.read(tupleLen*2))
                Bits_Per_Sample.append(aTuple)
            else:
                Bits_Per_Sample.append(IFD_DE_List[i][258])
            Color_Space.append(IFD_DE_List[i][262])
            Compression_Type.append(IFD_DE_List[i][259])
            #计算图像部分大小
            if (279 in IFD_DE_List[i]):
                #如果没有采用瓦片切片存储
                stripNum = IFD_DE_List[i][279][1]  #获取条带大小列表的长度
                if (stripNum == 1):
                    #条带大小可能存为单个值
                    Pixel_Byte_Count = IFD_DE_List[i][279][2]
                else:
                    stripByteOffset = IFD_DE_List[i][279][2]  #获取条带大小列表的偏移量
                    stripType = Tiff_Field_Type_Dict[IFD_DE_List[i][279][0]]  #获取条带值的存储类型
                    stripLength = Tiff_Field_Length_Dict[IFD_DE_List[i][279][0]]  #获取条带值的字节长度
                    fo.seek(stripByteOffset,0)
                    stripByteTuple = struct.unpack(str(stripNum)+stripType,fo.read(stripNum*stripLength))  #读取条带大小列表
                    for j in range(stripNum):
                        Pixel_Byte_Count += stripByteTuple[j]  #累加条带大小列表中的值即可得到图像部分大小
            elif (325 in IFD_DE_List[i]):
                #如果采用了瓦片切片存储
                tileNum = IFD_DE_List[i][325][1]  #获取瓦片大小列表的长度
                if (tileNum==1):
                    #瓦片大小可能存为单个值
                    Pixel_Byte_Count = IFD_DE_List[i][325][2]
                else:
                    tileByteOffset = IFD_DE_List[i][325][2]  #获取瓦片大小列表的偏移量
                    tileType = Tiff_Field_Type_Dict[IFD_DE_List[i][325][0]]  #获取瓦片值的存储类型
                    tileLength = Tiff_Field_Length_Dict[IFD_DE_List[i][325][0]]  #获取瓦片值的字节长度
                    fo.seek(tileByteOffset,0)
                    tileByteTuple = struct.unpack(str(tileNum)+tileType,fo.read(tileNum*tileLength))  #读取瓦片大小列表
                    #print(tileByteTuple)
                    for j in range(len(tileByteTuple)):
                        Pixel_Byte_Count += tileByteTuple[j]  #累加瓦片大小列表中的值即可得到图像部分大小
            else:
                print("存在未识别的tif存储结构, tif压缩率计算有误!")
                return -1.0

        #计算图像部分估计大小
        for i in range(Band_Count):
            if (type(Bits_Per_Sample[i])==tuple):
                Bits_Per_Sample_Sum = 0
                for j in Bits_Per_Sample[i]:
                    Bits_Per_Sample_Sum += j
            else:
                Bits_Per_Sample_Sum = Bits_Per_Sample[i]
            Estimated_Pixel_Byte_Count += Raster_Width[i] * Raster_Length[i] * (Bits_Per_Sample_Sum/8)

        Compression_Rate = Pixel_Byte_Count / Estimated_Pixel_Byte_Count
        fo.close()
        return Compression_Rate
    else:
        print("错误! 输入的不是tif文件, 无法计算压缩率!")
        return -1.0

def calPngPA(path0: str, path1: str, colorList: list) -> float:
    '''
    计算两张png图片综合前后的位置准确度
    '''
    imgRGB0 = cv2.imread(path0,cv2.IMREAD_COLOR)
    imgRGB1 = cv2.imread(path1,cv2.IMREAD_COLOR)
    breakNum = len(colorList)
    IoUList = []
    posAcc = -1

    # 考虑等级,要按照BGR值范围提取各等级图像,分别计算IoU
    # Color_List中的元素必须为numpy.array
    Color_List = []
    for i in range(breakNum):
        Color_List.append(np.array(colorList[i]))

    # 依次计算各等级IoU
    for i in range(breakNum):
        img0Lv = cv2.inRange(imgRGB0, Color_List[i], Color_List[i])
        img1Lv = cv2.inRange(imgRGB1, Color_List[i], Color_List[i])
        IoULv = countIoU(cv2.bitwise_and(img0Lv,img1Lv), cv2.bitwise_or(img0Lv,img1Lv))
        IoUList.append(IoULv)

    # #查看过程图像
    # cv2.imshow(path1,img1Lv3)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 确定权重, 确保最高的两个等级占比达到0.7
    if (breakNum == 5):
        weightList = [0.1, 0.1, 0.1, 0.3, 0.4]
    else:
        weightList = []
        for i in range(breakNum-2):
            weightList.append(0.3/(breakNum-2))
        weightList.append(0.3)
        weightList.append(0.4)

    # print(weightList,IoUList)
    posAcc = np.dot(np.array(weightList), np.array(IoUList))
    return posAcc

def countIoU(inter,union) -> float:
    '''
    从交集图像和并集图像中计算IoU
    '''
    x = cv2.countNonZero(inter)
    y = cv2.countNonZero(union)
    if (y==0):
        return 1.0
    else:
        return x/y

#模块测试用
if __name__ == '__main__':
    os.chdir(r'C:\Postgraduate_Work\地表异常遥感预警知识即时表达\资源占用度量\示例数据\tiff_compressed')
    # os.chdir(r'C:\Postgraduate_Work\CodeFolder\20231202_allData\no_trend\palu-tsunami\inf')
    file_name_list = os.listdir()
    tif_file_name_list = []
    for i in file_name_list:
        fext = i.split('.')[-1].lower()
        if (fext=='tif'):
            tif_file_name_list.append(i)

    for i in tif_file_name_list:
        print(i, calTifCR(i))