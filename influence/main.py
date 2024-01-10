import slice
import merge
import time
import json
import evaluateImage as eval
import generalisation as gene
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import psutil
import function as func


def inf_building_main(inf_building_path, ResultFolderPath, TempFolderPath, cGAN_path):
    res = {}
    dname = inf_building_path.split('\\')[-1].split('.')[0]
    start = time.time()
    start = time.time()
    params = slice.colorTif(inf_building_path, TempFolderPath)
    slice_result = slice.slice1(params, TempFolderPath)
    start1 = time.time()
    gene.process_generalisation(slice_result[0], cGAN_path)
    # 第一个参数是存放综合结果的路径，第二个和第三个参数是tif的额行列数，第四个是临时路径。第五个是原始图像的路径
    merge_result = merge.mergeic(
        cGAN_path, slice_result[2], slice_result[3], TempFolderPath, inf_building_path)
    mergepath = merge.addCoor(
        merge_result, inf_building_path, ResultFolderPath)
    end = time.time()
    print(end-start)
    print(end-start1)
    # 计算凸包
    hull_tif = TempFolderPath + '\\hull.tif'
    hull_shp = ResultFolderPath + '\\g_iflres\\' + dname + '_mask.shp'
    print(hull_shp)
    img, m, n, geotrans, proj = func.readImg(inf_building_path)
    img_ch = func.calConvexHull(img)
    func.WriteTifImg(hull_tif, proj, geotrans, img_ch)
    func.TiftoShp(hull_tif, 'Value', hull_shp)
    # end = time.time()
    runtime = end - start
    eval_list = [-1, -1, -1, 1, 1]
    # eval_list = eval.evaluateLvTif(params[0], merge_result, temp)
    res["disasterTime"] = inf_building_path.split(
        '\\')[-1].split('.')[0].split('_')[0]
    res["runtime"] = runtime
    res['cpuUtilization'] = psutil.virtual_memory().percent  # CPU占用率（%）
    res["edgeDensity"] = eval.calPngED(params[0])
    res["edgeDensityImprovementRate"] = (
        res["edgeDensity"] - eval.calPngED(merge_result))/res["edgeDensity"]
    res["positionalAccuracy"] = eval.calPngPA(params[0],merge_result,eval.Color_List_Influence_4)
    res["comp"] = round(eval.calTifCR(inf_building_path), 2)
    res["compImprovement"] = (eval.calTifCR(
        inf_building_path) - eval.calTifCR(mergepath))/eval.calTifCR(
        inf_building_path)
    res["sizeBefore"] = round(os.path.getsize(
        inf_building_path) / 1024 / 1024, 2)  # 综合前数据大小（MB）
    res["sizeAfter"] = round(os.path.getsize(
        mergepath) / 1024 / 1024, 2)  # 综合后数据大小（MB）
    res["showTif"] = mergepath
    return res, hull_shp


if __name__ == '__main__':
    path = r"C:\Users\nana_\Desktop\shpdata\part_palu\palu_part.tif"
    temppath = r"F:\Unusual\report\1223\palup_inf"
    resultpath = r"F:\Unusual\report\1223\palup_inf"
    cGAN_path = r"F:\Unusual\Group3strenth1207\report\system_show\palu_tsunami\new"
    result, hull_shp = inf_building_main(path, resultpath, temppath, cGAN_path)
    print(result)
    print(hull_shp)
    print("finished")
