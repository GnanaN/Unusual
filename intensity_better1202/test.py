import concurrent.futures
import time
import jenkspy
import cv2
import numpy as np
import function as func
from matplotlib import pyplot as plt
import rasterio
# plantpath1 = r"F:\Unusual\Group3strenth1207\group3_eg\glass-fire\202010101235_inf_plant.tif"
# plantpath2 = r"F:\Unusual\Group3strenth1207\group3_3857\midwest-flooding\inf\201908231132_inf_plant.tif"
# buildpath2 = r"F:\Unusual\Group3strenth1207\group3_3857\midwest-flooding\inf\201908231132_inf_building.tif"
# buildpath3 = r"F:\Unusual\Group3strenth1207\group3_3857\palu-tsunami\inf\201811231323_inf_building.tif"

# planthull = r'F:\Unusual\Group3strenth1207\result\inf_temp\plant_hull1.tif'
# planthul2 = r'F:\Unusual\Group3strenth1207\result\inf_temp\plant_hull2.tif'
# buildhul2 = r'F:\Unusual\Group3strenth1207\result\inf_temp\build_hull2.tif'
# buildhul3 = r'F:\Unusual\Group3strenth1207\result\inf_temp\build_hull3.tif'

# plantshp1 = r'F:\Unusual\Group3strenth1207\result\inf_temp\glass_fire_plant.shp'
# plantshp2 = r'F:\Unusual\Group3strenth1207\result\inf_temp\flood_plant.shp'
# buildshp2 = r'F:\Unusual\Group3strenth1207\result\inf_temp\flood_build.shp'
# buildshp3 = r'F:\Unusual\Group3strenth1207\result\inf_temp\palu_build.shp'


# img, m, n, geotrans, proj = func.readImg(plantpath1)
# img1 = func.calConvexHull(img)
# func.WriteTifImg(planthull, proj, geotrans, img1)
# func.TiftoShp(planthull,'Value',plantshp1)

# img, m, n, geotrans, proj = func.readImg(plantpath2)
# img1 = func.calConvexHull(img)
# func.WriteTifImg(planthul2, proj, geotrans, img1)
# func.TiftoShp(planthul2,'Value',plantshp2)
# print('plant done')

# img, m, n, geotrans, proj = func.readImg(buildpath2)
# img1 = img.astype('float32')
# img1 = func.calConvexHull(img1)
# func.WriteTifImg(buildhul2, proj, geotrans, img1)
# func.TiftoShp(buildhul2, 'Value', buildshp2)
# print('build done')

# img, m, n, geotrans, proj = func.readImg(buildpath3)
# img1 = img.astype('float32')
# img1 = func.calConvexHull(img1)
# func.WriteTifImg(buildhul3, proj, geotrans, img1)
# func.TiftoShp(buildhul3, 'Value', buildshp3)


# 3-5级凸包生成代码
# import os
# path = r'F:\Unusual\Group3strenth1207\result\2group3_1212\data2group3'
# filelist = os.listdir(path)
# for file in filelist:
#     tiflist = os.listdir(os.path.join(path,file))
#     for tif in tiflist:
#         # 找到每一个文件夹下的tif文件
#         if (tif.split('.')[-1] == 'tif' and tif.split('.')[0].split('_')[-1] == 'res'):
#             print(tif)
#             # 保存凸包tif 的路径
#             savetif = r'F:\Unusual\Group3strenth1207\result\2group3_1212\temp\res_tif' + '\\img_hull.tif'
#             shp = os.path.join(path,file,tif.split('_')[0] + '_str_mask.shp')
#             if os.path.exists(savetif):
#                 os.remove(savetif)
#             if os.path.exists(shp):
#                 os.remove(shp)
#             tifpath = os.path.join(path,file,tif)
#             img, m, n, geotrans, proj = func.readImg(tifpath)
#             for i in range(3,6):
#                 savetif = r'F:\Unusual\Group3strenth1207\result\2group3_1212\temp\res_tif' + '\\img_hull.tif'
#                 if os.path.exists(savetif):
#                     os.remove(savetif)
#                 img_mask = np.zeros_like(img)
#                 img_mask[img == i] = 1
#                 shp1 = os.path.join(path, file, tif.split('_')[0] + '_str_mask_'+ str(i) + '.shp')
#                 img_ch = func.calConvexHull(img_mask)
#                 func.WriteTifImg(savetif, proj, geotrans, img_ch)
#                 func.TiftoShp(savetif, 'Value', shp1)


# path1 = r"F:\Unusual\Group3strenth1207\temp\202003281832_str_reclass_mask.tif"
# path2 = r"F:\Unusual\Group3strenth1207\result\fire_scmuli\202003281832_str_show.tif"

# img1,_,_,_,_ = func.readImg(path1)
# img2,_,_,_,_ = func.readImg(path2)


# def cal_patch_num(img):
#   img = img.astype('uint8')
#   for i in range(1, 6):
#     img_copy = np.zeros_like(img)
#     img_copy[img == i] = 1
#     num_label, labels, stats, centroids = cv2.connectedComponentsWithStats(
#         img_copy, connectivity=8)
#     print('等级%d的斑块数为%d' % (i, num_label-1))
# cal_patch_num(img1)
# cal_patch_num(img2)

import evaluateTif as eval
path = r"F:\Unusual\Group3strenth1207\result\pic\glass_plant_orgin.jpg"

print(eval.calCR(path))
print(eval.calED(path))


