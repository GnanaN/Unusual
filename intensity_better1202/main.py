import numpy as np
from osgeo import gdal, ogr, osr
import ConverxHull as ch
import os
import cv2
from skimage import io
import function as func
from matplotlib import pyplot as plt
import time
import evaluateTif as evat
from scipy import stats
import psutil
import pandas as pd
import triggerGeneralize as trig

plt.rcParams['font.sans-serif'] = ['SimHei']

def main(dataFloderPath, ResultfloderPath, TempfloderPath):
    index= []
    disatername = dataFloderPath.split('\\')[-1]
    ResultfloderPath = os.path.join(ResultfloderPath, disatername)    # --> 初始化文件夹
    if os.path.exists(ResultfloderPath):
        pass
    else:
        os.mkdir(ResultfloderPath)
    func.floderPrepare(ResultfloderPath, TempfloderPath)
    # 遍历出所有需要处理的tif文件
    tiflist = func.FindData(dataFloderPath)                           # -->读取数据
    for tif in tiflist:
        name = (tif.split('\\')[-1].split('.')[0])
        if name.split('_')[-1] == 'str':
            mask_tif_path =  os.path.join(
                TempfloderPath, name + '_mask.tif')
            mask_shp_path = os.path.join(
                TempfloderPath, name + '_mask.shp')
            hull_tif_path  = os.path.join(
                TempfloderPath, name + '_hull.tif')
            hull_shp_path = os.path.join(
                ResultfloderPath, name + '_mask.shp')
            core_reclass_tif_path = os.path.join(
                TempfloderPath, name + '_reclass_mask.tif')
            result_tif_path =  os.path.join(
                ResultfloderPath, name + '_res.tif')
            result_shp_path = os.path.join(
                ResultfloderPath, name + '_res.shp')
            show_tif_path = os.path.join(
                ResultfloderPath, name + '_show.tif')
            img_org, m, n,geotrans, proj = func.readImg(tif)
            condition = trig.need_generalize(img_org)                            #-->判断是否需要综合
            img_org ,m,n,geotrans, proj = func.readImg(tif)
            # 如果你的-3.402823e+38值被视为无效数据，并且你想在计算最小值时包含这些值，你需要先将这些值转换为有效的数值
            img = np.where(img_org < -3.402823e+38, -99, img_org)
            index.append(os.path.getsize(tif)/1024/1024)
            # 判断数据是连续的还是离散的
            img_temp = np.zeros_like(img)
            img_temp[img>-10] = 1
            img_temp = img_temp.astype(np.uint8)
            num_labels, labels, _, centers = cv2.connectedComponentsWithStats(
                img_temp, connectivity=4, ltype=cv2.CV_32S)
            # 这里需要加一部触发机制判断的步骤,如果需要综合那么就是下面的步骤，否则就直接转shp
            start = time.time()
            if condition:
                if num_labels >2 or (-99 not in img):                            #-->判断数据类型并综合
                    print('是我')
                    # 这里数据有两种一种是有Nodata的，一种是没有Nodata的，把前者转换为后者
                    img_blur, k_size = func.GaussBlur(img, 0.005)  # 高斯模糊
                    img_blur = img
                    kmeans_labels = func.kCluster(img_blur, 2)  # kmeans聚类
                    kmeans_img = func.TransToArray(
                        img_blur, m, n, kmeans_labels)  # 将聚类结果转换为图像
                    # 这里有个问题，kmeans_img是三维的，但是只有一个通道，所以需要转换为二维
                    kmeans_img = kmeans_img[:, :, 0]
                    # 提取轮廓
                    outline_img = func.OpenClose(kmeans_img, func.designKernel(
                        func.calDirection(kmeans_img), k_size))  # 开闭运算
                    outline_img = outline_img.astype(np.uint8)
                    # k_size = 3
                else:
                    outline_img = np.zeros_like(img)
                    outline_img[img > -1] = 1
                    outline_img = outline_img.astype(np.uint8)
                    img = func.ChangeBackgValue(img)
                    img_blur, k_size = func.GaussBlur(img, 0.005)
                    k_size = 25
                    img_blur = img
                # plt.imshow(outline_img)
                # plt.show()
                '''
                # mask转tif
                func.WriteTifImg(mask_tif_path, proj, geotrans, outline_img)
                # mask转shp
                func.TiftoShp(mask_tif_path, 'Value', mask_shp_path)
                img_ch = func.calConvexHull(outline_img)
                # 凸包转tif
                func.WriteTifImg(hull_tif_path, proj, geotrans,img_ch)
                # 凸包转shp
                func.TiftoShp(hull_tif_path, 'Value' ,hull_shp_path)
                '''
                #用掩膜提取核心区域
                core_img = func.imgbymask(img_blur,outline_img)
                core_reclass = func.reclass(core_img,5,'12')    # 自然断点法
                # core_reclass = func.reclass(core_img, 5)    #等间隔

                mode = stats.mode(core_reclass[core_reclass > 0],keepdims=True)
                func.WriteTifImg(core_reclass_tif_path, proj, geotrans,core_reclass)
                #开始综合
                closed_img = func.LayerProcess(core_reclass, outline_img, k_size)
                grade3 = np.zeros_like(closed_img)
                grade3[closed_img > 2] = 1

                img_ch = func.calConvexHull(grade3)

                # 凸包转tif
                func.WriteTifImg(hull_tif_path, proj, geotrans, img_ch)
                # 凸包转shp
                func.TiftoShp(hull_tif_path, 'Value', hull_shp_path)
                show_img = func.LayerProcess2_noremove(core_reclass,outline_img, k_size)
                func.WriteTifImg(result_tif_path, proj, geotrans,closed_img)
                func.WriteTifImg(show_tif_path, proj, geotrans, show_img)
                func.TiftoShp(result_tif_path, 'Value', result_shp_path)
                # 删除板块
                func.delete_shp(result_shp_path)
                func.edit_value(hull_shp_path,int(mode.mode[0]))
                eval_list = evat.evaluateLvTif(
                    core_reclass_tif_path, result_tif_path, TempfloderPath)
            else:
                print(disatername+'需要综合')
                if len(np.unique(img))<10:
                    # 将原图保存到result文件夹中，用于展示
                    img = func.ChangeBackgValue_noge(img)
                    mode = stats.mode(img[img > 0],keepdims=True)
                    func.WriteTifImg(result_tif_path, proj, geotrans, img, proj)
                    # 将原图转换成shp,用于语义提取
                    func.TiftoShp(result_tif_path, 'Value', result_shp_path)
                    # 外部轮廓
                    img_ch = func.calConvexHull(img)
                    # 凸包转tif
                    func.WriteTifImg(hull_tif_path, proj, geotrans, img_ch, proj)
                    # 凸包转shp
                    func.TiftoShp(hull_tif_path, 'Value', hull_shp_path)
                else:
                    img1 = np.zeros_like(img)
                    img1 = np.where(img > 0, img, img1)
                    img = func.imgbymask(img1)
                    img = func.reclass(img, 5)
                    mode = stats.mode(img[img > 0],keepdims=True)
                    func.WriteTifImg(result_tif_path, proj, geotrans, img, proj)
                    # 将原图转换成shp,用于语义提取
                    func.TiftoShp(result_tif_path, 'Value', result_shp_path)
                    # 外部轮廓
                    img_ch = func.calConvexHull(img)
                    # 凸包转tif
                    func.WriteTifImg(hull_tif_path, proj, geotrans, img_ch, proj)
                    # 凸包转shp
                    func.TiftoShp(hull_tif_path, 'Value', hull_shp_path)
                func.edit_value(hull_shp_path, int(mode.mode[0]))
                func.delete_shp(result_shp_path)
                eval_list = evat.evaluateLvTif(result_tif_path,result_tif_path,TempfloderPath)
    end = time.time()
    runtime = end-start
    print(runtime)
    index.append(psutil.virtual_memory().percent)
    index.extend(eval_list)
    index.append(runtime)
    index.append(psutil.cpu_percent(interval=runtime))
    index.append(os.path.getsize(result_tif_path)/1024/1024)
    index.append(result_tif_path)
    return index


path = r"F:\Unusual\datafromwcx1226\str"
tempath = r"F:\Unusual\datafromwcx1226\gene\temp"
resultpath = r"F:\Unusual\datafromwcx1226\gene\result"
print(main(path, resultpath, tempath))

# continus = ['rain_qilianshan']
# # continus = ['calfornia2', 'calfornia3', 'fire_scmuli', 'rain_qilianshan', 'taihu1']
# # disperse = ['calfornia1', 'henan_snow', 'hubei_coast']
# # disperse1 = ['xinjiang_earthquack','xizang_drought']
# path = r"F:\Unusual\All_Data_Test1114\all_distaer_1114\NoTrend"
# indexl = []
# for c in continus:
#     indexlist = main(os.path.join(path, c), resultpath, tempath)
#     indexl.append(indexlist)
#     print(indexl)
# # for d in disperse:
# #     indexlist = main(os.path.join(path, d), resultpath, tempath)
# #     indexl.append(indexlist)
# #     print(indexl)
# # 把indexl写入excel,添加列名
# indexl = pd.DataFrame(indexl)
# indexl.columns = [ '综合前数据大小(MB)', '运行内存占用率(%)', '边缘密度', '边缘密度改善率',
#                   '位置准确度', '压缩率', '压缩率改善率', '运行时间', 'cpu占用率(%)', '综合后数据大小(MB)','灾害名']
# indexl.to_excel(r'F:\Unusual\All_Data_Test1114\all_distaer_1114\1206\Notrend_result\index.xlsx')

# path = r"F:\Unusual\Group3strenth1207\group3_3857\glass-fire\str\202010101235_str.tif"
# c = trig.need_generalize(path)
# print(c)