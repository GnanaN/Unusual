'''
综合的触发机制
吕开来
20231208
'''
import os
import numpy as np
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import evaluateTif

#2023年底示例采用的colorList，颜色为BGR格式
Color_List_Strength_3 = [[0,168,56],[0,255,255],[0,0,255]]
Color_List_Influence_4 = [[0,168,56],[0,209,139],[0,128,255],[0,0,255]]
Color_List_Strength_5 = [[0,168,56],[0,209,139],[0,255,255],[0,128,255],[0,0,255]]
Color_List_Strength_10 = [[0,97,0],[0,128,60],[0,161,107],[0,196,164],[0,235,223],[0,234,255],[0,187,255],[0,145,255],[0,98,255],[0,34,255]]

def need_generalize(img, breakList: list = None, colorList: list = None) -> bool:
    '''
    判断栅格图像是否需要综合

    Parameters
    ----------
    path : str
        tif栅格的路径
    breakList : list
        分类断点值列表，其中应包括最大值和最小值
    colorList : list
        色彩映射表，各颜色应为[B,G,R]格式

    Returns
    -------
    result : bool
        是否需要综合
    '''

    #打开原始tif栅格
    imgRaw = img
    #检查分类断点值列表和色彩映射表，若任一未提供则应用默认分类和色彩映射表
    if (breakList==None or colorList==None):
        # 判断是等级栅格还是连续栅格
        lvList = np.unique(imgRaw)
        uniNum = len(lvList) - 1  #减去代表NoData的值
        if (uniNum > 10):
            # 唯一值数量大于10, 判断为连续栅格
            breakList = getBreakList(ChangeBackgValue(imgRaw), 5)
            colorList = Color_List_Strength_5
        elif (int(lvList[-1]) > 5):
            # 等级栅格10级
            breakList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            colorList = Color_List_Strength_10
        else:
            # 等级栅格5级
            breakList = [0, 1, 2, 3, 4, 5]
            colorList = Color_List_Strength_5
    #若分类断点值列表和色彩映射表内的等级数不一致，则报错
    elif (len(colorList) != len(breakList)-1):
        print("ERROR: breakList与colorList的等级数不一致。综合触发检验将被跳过。")
        return False

    #基于分类断点值列表进行重分类和色彩渲染
    breakList[0] = breakList[0] - 0.0001
    imgBGR = np.zeros((imgRaw.shape[0], imgRaw.shape[1], 3), dtype="uint8")
    for i in range(len(breakList)-1):
        imgBGR[(imgRaw[:,:] > breakList[i]) & (imgRaw[:,:] <= breakList[i+1])] = colorList[i]

    # # 检查色彩渲染结果
    # cv2.imshow('imgBGR',imgBGR)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #正式进行综合触发检验
    #1、资源占用检验
    Size_Threshold = 1024000  #资源占用阈值，单位为Byte
    cv2.imwrite('sizeTest.png',imgBGR)
    # print(os.path.getsize('sizeTest.png')/1024)
    if (os.path.getsize('sizeTest.png') > Size_Threshold):
        os.remove('sizeTest.png')
        return True

    #2、压缩率检验
    CR_Threshold = 0.05  #压缩率阈值，单位为%
    # print(compress.cal_png_compression_rate('sizeTest.png'))
    if (evaluateTif.calCR('sizeTest.png') > CR_Threshold):
        os.remove('sizeTest.png')
        return True

    #3、清晰度检验
    Leg_Threshold = 0.03  #边缘密度阈值，单位为%
    # print(inds.calLeg('sizeTest.png'))
    if (evaluateTif.calED('sizeTest.png') > Leg_Threshold):
        os.remove('sizeTest.png')
        return True

    #4、图斑数量检验
    VecNum_Threshold = 200  #图斑数量阈值，单位为个
    vecNum = 0
    for i in range(len(colorList)):
        colorI = np.array(colorList[i])
        imgLvI = cv2.inRange(imgBGR, colorI, colorI)
        contours, hierarchy = cv2.findContours(imgLvI,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if (i != range(len(colorList))[-1]):
            vecNum += int(len(contours)/2)
        else:
            vecNum += len(contours)
    # print(vecNum)
    if (vecNum > VecNum_Threshold):
        os.remove('sizeTest.png')
        return True

    os.remove('sizeTest.png')
    return False

# 改变背景值
def ChangeBackgValue(img):
    img[img < -100] = 9999
    img_min = np.nanmin(img)
    img[img == 9999] = img_min
    img[img > 9999] = img_min
    img[np.isnan(img)] = img_min

    # print('背景值修改成功')
    return img

# 等间距分级
def getBreakList(img, break_num):
    '''
    等间隔分级方法
    '''
    ChangeBackgValue(img)
    max_value = np.max(img)
    min_value = np.min(img)

    breaks = [min_value]
    step = (max_value - min_value) / break_num
    for i in range(break_num - 1):
        breaks.append(min_value + (i + 1) * step)
    breaks.append(max_value)
    # print("breaks",breaks)
    return breaks

#模块测试用
# if __name__ == '__main__':
    # os.chdir(r'C:\Postgraduate_Work\CodeFolder\20231202_allData')
    # print(need_generalize(r'.\no_trend\ca3\201811081623_str.tif'))  #True
    # print(need_generalize(r'.\no_trend\xinjiang_earthquack\tif1.tif'))  #False
    # print(need_generalize(r'.\no_trend\xizang_drought\tif2.tif'))  #False
    # print(need_generalize(r'.\trend\taihu\202005090826_str.tif'))  #True


