import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from torchray.attribution.grad_cam import grad_cam
import torch
import torchvision
from torchvision import transforms

# Suppress warnings due to pytorch version source change
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)
#User Defined Parameters
os.chdir('./') # Change location of root directory
root = os.getcwd()

model_path = 'D:/NeurIPS CLEAR/'
original_images_path = 'D:/NeurIPS CLEAR/Synthetic Data/diseased/'
GAN_images_path = 'D:/NeurIPS CLEAR/Synthetic Data/gen_healthy/'
GradCAM_Densenet_path = 'D:/CLEAR AIC Results/Synthetic/GradCam/Densenet/'
GradCAM_VGG_path = 'D:/CLEAR AIC Results/Synthetic/GradCam/VGG/'
Extremal_Densenet_path = 'D:/CLEAR AIC Results/Synthetic/Extremal/Densenet/'
Extremal_VGG_path = 'D:/CLEAR AIC Results/Synthetic/Extremal/VGG/'
CLEAR_Densenet_path = 'D:/NeurIPS CLEAR/CLEAR Heatmaps/Synthetic/Densenet full/'
CLEAR_VGG_path = 'D:/NeurIPS CLEAR/CLEAR Heatmaps/Synthetic/VGG full/'
LIME_Densenet_path = 'D:/NeurIPS CLEAR/Lime Heatmaps/Synthetic/Densenet/'
LIME_VGG_path = 'D:/NeurIPS CLEAR/Lime Heatmaps/Synthetic/Densenet/'
Annotated_Image_path = 'D:/NeurIPS CLEAR/Synthetic Data/pointing/'
anno = os.listdir(Annotated_Image_path) # Change location for Xpert Annotated Images
mdl = 'Densenet' # Choice: VGG or Densenet (Case Sensitive)
mtd = 3 # 0 - CLEAR, 1 - GradCAM, 2 - Extremal, 3 - LIME

if not ((mdl == 'VGG') or (mdl == 'Densenet')):
    print('Please select either VGG or Densenet (Case sensitive).')

# Dice Score function with intersection calculation
def dice(img, img2):
        if img.shape != img2.shape:
            raise ValueError("Shape mismatch: img and img2 must have to be of the same shape.")
        else:
            
            lenIntersection=0
            
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if ((img[i][j] ==1) & (img2[i][j]==1)):
                        lenIntersection+=1
             
            lenimg=np.sum(img)
            lenimg2=np.sum(img2)

            dice = (2. * lenIntersection  / (lenimg + lenimg2))
        return lenIntersection, lenimg, lenimg2, dice

lst = []
totalgt = 0
colour = ['Reds', 'Greens', 'Blues', 'Purples', 'Greys']
bstdf = pd.DataFrame(columns=['fname','iou'])
for im in range(len(anno)):
  try:
    gcount = 0 # Initialize # GT segments to zero
    glst = [] # Initialize valid gt segments after contour fill
    ccount = 0 # Initialize # CLEAR segments to zero
    clst = [] # Initialize valid CLEAR segments after contour fill
    nseg = 3 # Number of segments 

    fname = anno[im].split('.')[0][8:]
    # print(fname)
    gt = cv2.imread(Annotated_Image_path + anno[im], 0)
    gtd = cv2.resize(cv2.imread(original_images_path + 'diseased'+ fname+'.png'), (224,224))
    gt_img = gt.copy()
    contours, hier = cv2.findContours(gt_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    gtblob = np.zeros((224,224))
    cv2.drawContours(gtblob, contours, -1, 1, -1)
  
    if mtd == 0:
        # Generate all separate CLEAR blobs:
        if mdl == 'Densenet':
            CLEAR_file = CLEAR_Densenet_path + 'CLEAR_VGG_point_heatmap_' + fname + '.csv'
        else:
            CLEAR_file = CLEAR_VGG_path + 'CLEAR_VGG_point_heatmap_' + fname + '.csv'
        clr = np.genfromtxt(CLEAR_file, delimiter=",")   # Change directory prefix to the heatmap location for clear
        clr_norm = (clr - clr.min())/(clr.max() - clr.min())

        pos = np.sort(np.unique(clr_norm))[::-1]
        pos = pos[pos>0.7]
    elif mtd == 1:
        # Generate all separate Grad-CAM blobs:
        if mdl == 'Densenet':
            Grad_file = GradCAM_Densenet_path + 'GradCam_Densenet_heatmap_' + fname + '.csv'
        else:
            Grad_file = GradCAM_VGG_path + 'GradCam_VGG_heatmap_' + fname + '.csv'
        gcam = np.genfromtxt(Grad_file, delimiter=",")
        gcam_norm = (gcam - gcam.min())/(gcam.max() - gcam.min())
        gcam_224_mask = np.where(gcam_norm >=0.90, 1,0)
        gcam_224_2nd = np.where((gcam_norm < 0.90) & (gcam_norm >=0.80), 1,0)
        gcam_224_3rd = np.where((gcam_norm < 0.80) & (gcam_norm >=0.70), 1,0)

        gcammask = [gcam_224_mask, gcam_224_2nd, gcam_224_3rd]
        pos = [0.95, 0.85, 0.75]
    elif mtd == 2:
        # Generate all separate extremal blobs:
        if mdl == 'Densenet':
            Extremal_file = Extremal_Densenet_path + 'Extremal_Densenet_heatmap_' + fname+ '.csv'
        else:
            Extremal_file = Extremal_VGG_path + 'Extremal_Densenet_heatmap_' + fname+ '.csv'
        ext = np.genfromtxt(Extremal_file, delimiter=",")
        ext_norm = (ext - ext.min())/(ext.max() - ext.min())
        ext_224_mask = np.where(ext_norm >=0.90, 1,0)
        ext_224_2nd = np.where((ext_norm < 0.90) & (ext_norm >=0.80), 1,0)
        ext_224_3rd = np.where((ext_norm < 0.80) & (ext_norm >=0.70), 1,0)

        extmask = [ext_224_mask, ext_224_2nd, ext_224_3rd]
        pos = [0.95, 0.85, 0.75]
    elif mtd == 3:
        # Generate all separate LIME blobs:
        if mdl == 'Densenet':
            LIME_file = LIME_Densenet_path + 'LIME_Densenet_heatmap_' + fname + '.csv'
        else:
            LIME_file = LIME_VGG_path + 'LIME_VGG_heatmap_' + fname + '.csv'
        lime = np.genfromtxt(LIME_file, delimiter=",")
        lime_norm = (lime - lime.min())/(lime.max() - lime.min())

        pos = np.sort(np.unique(lime_norm))[::-1]
        pos = pos[pos>0.7]

    fig = plt.figure(figsize=(5,5))
    plt.imshow(cv2.resize(gt, (224,224), interpolation = cv2.INTER_AREA))

    iou_best = []
    totalgt+=1
    topj = np.min([len(pos),4])

    if mtd == 0:
        xaiblob = np.where(clr_norm >= pos[topj - 1], 1, 0)
    elif mtd == 1:
        gcamm = np.zeros((224,224), dtype='uint8')
        for j in range(topj):
            gcamm = cv2.bitwise_or(gcamm, gcammask[j].astype('uint8'))
        xaiblob = gcamm
    elif mtd == 2:
        extm = np.zeros((224,224), dtype='uint8')
        for j in range(topj):
            extm = cv2.bitwise_or(extm, extmask[j].astype('uint8'))
        xaiblob = extm
    elif mtd == 3:
        xaiblob = np.where(lime_norm >= pos[topj - 1], 1, 0)
    segcheck = np.sum(xaiblob)/50176
    if segcheck >=0.5:
      print('Check background')
    intsect,a1, a2, _ = dice(np.uint8(gtblob), np.uint8(xaiblob))
    print(intsect, a1, a2)
    iou = intsect/(a1+a2-intsect)
    plt.imshow(xaiblob, cmap=colour[0], alpha=0.4)
    lst = [fname, iou]
    bstdf = bstdf.append(pd.Series(lst, index=['fname', 'iou']), ignore_index=True)
    print(pd.Series(lst, index=['fname', 'iou']))
    plt.axis('off')
    plt.close()
    print('--------------------')
  except:
    pass
        
print('Total images evaluated: ', totalgt)
print('Average IOU Score: ', np.round(bstdf['iou'].mean(),3))
# bstdf.to_csv('./ext_densenet_iou_xpert.csv') # Save specific IOU result as csv file