import cv2
import numpy as np
import evaluateTif as eval
import os
cv2.setUseOptimized(True)
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
from skimage import io
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

path0 = r"F:\Unusual\report\1223\palup_inf\g_ifltemp\RC_color.png"
path1 = r"F:\Unusual\report\1223\palup_inf\g_ifltemp\merge.png"
path2 = r"C:\Users\nana_\Desktop\shpdata\part_palu\palu_gene.png"
temp = r'F:\Unusual\report\1223\flood_inf\temp\g_ifltemp'

# img = io.imread(path0)
ED = eval.calED(path1)
# CR = eval.calCR(path1)
PA = eval.calPA(path0, path1, eval.Color_List_Influence_4)
print(ED,PA)
EDv = eval.calED(path2)
# CR = eval.calCR(path1)
PAv = eval.calPA(path0, path2, eval.Color_List_Influence_4)
print(EDv,PAv)

