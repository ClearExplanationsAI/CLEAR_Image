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
original_images_path = 'D:/NeurIPS CLEAR/CheXpert Data/1_pleural_effusion/'
GAN_images_path = 'D:/NeurIPS CLEAR/CheXpert Data/0_healthy/'
GradCAM_Densenet_path = 'D:/CLEAR AIC Results/Xrays/GradCam/Densenet/'
GradCAM_VGG_path = 'D:/CLEAR AIC Results/Xrays/GradCam/VGG/'
Extremal_Densenet_path = 'D:/CLEAR AIC Results/Xrays/Extremal/Densenet/'
Extremal_VGG_path = 'D:/CLEAR AIC Results/Xrays/Extremal/VGG/'
CLEAR_Densenet_path = 'D:/NeurIPS CLEAR/CLEAR Heatmaps/Xrays/Densenet full/'
CLEAR_VGG_path = 'D:/NeurIPS CLEAR/CLEAR Heatmaps/Xrays/VGG full/'
LIME_Densenet_path = 'D:/NeurIPS CLEAR/Lime Heatmaps/Xrays/Densenet/'
LIME_VGG_path = 'D:/NeurIPS CLEAR/Lime Heatmaps/Xrays/VGG/'
Annotated_Image_path = 'D:/NeurIPS CLEAR/Annotated Xrays/'
anno = os.listdir(Annotated_Image_path) # Change location for Xpert Annotated Images
org_diseased = original_images_path #Change location of Input Diseased Images for GradCAM. Please seek approval from the original author for CheXpert for original diseased images.
mdl = 'Densenet' # Choice: VGG or Densenet (Case Sensitive)
mtd = 0 # 0 - CLEAR, 1 - GradCAM, 2 - Extremal, 3 - LIME
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

# Pre-processing for GRADCAM model input
def Preprocess_Xray(input_image):
    preprocess = transforms.Compose([
        transforms.Resize(230),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    input_tensor = preprocess(input_image)
    return(input_tensor)

lst=[]
totalgt = 0
iouthresh = 0.95
colour = ['Reds', 'Blues', 'Greens','Purples', 'Greys']
bstdf = pd.DataFrame(columns=['fname','iou'])
for im in range(len(anno)):
  try:
    gcount = 0 # Initialize # GT segments to zero
    glst = [] # Initialize valid gt segments after contour fill
    ccount = 0 # Initialize # CLEAR segments to zero
    clst = [] # Initialize valid CLEAR segments after contour fill
    nseg = 3 # Number of segments 

    # fname = anno[im].rsplit('_',1)[0][7:]
    fname = anno[im].split('.')[0][7:]
    print(fname)
    gt = cv2.cvtColor(cv2.imread(os.path.join(Annotated_Image_path, anno[im])), cv2.COLOR_BGR2RGB)
    print(gt.shape)
    # Generate and create seperate ground truth blobs (Reds)
    gt_hsv = cv2.cvtColor(gt, cv2.COLOR_RGB2HSV)
    lower_red = np.array([0, 100, 0])
    upper_red = np.array([15, 255, 255])

    gt_red_mask = cv2.inRange (gt_hsv, lower_red, upper_red)
    if gt_red_mask.max() > 0:
      gt_red_norm = (gt_red_mask - gt_red_mask.min())/(gt_red_mask.max() - gt_red_mask.min())
    else:
      print("Red Mask not present")

    gt_red_224 = cv2.resize(gt_red_norm, (224,224), interpolation = cv2.INTER_AREA)
    ret, gt224 = cv2.threshold(gt_red_224, 0.5, 1, cv2.THRESH_BINARY)
    kernel3 = np.ones((3,3), np.uint8)
    kernel5 = np.ones((5,5), np.uint8)
    kernel7 = np.ones((7,7), np.uint8)
    gt224_dilate = cv2.dilate(gt224, kernel5, iterations=1)
    gt224_erode2 = cv2.erode(gt224_dilate, kernel5, iterations=1)
    ret, labels = cv2.connectedComponents(np.uint8(gt224_erode2))
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch =  255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = labeled_img[:,:,0]
    labeled_img[label_hue == 0] = 0
    for lbl in np.unique(labeled_img):
      segcnt = np.count_nonzero(np.where(labeled_img==lbl))
      if (segcnt < 20) | (segcnt > 0.5*224*224):
        labeled_img[labeled_img == lbl] = 0

    glst = list(np.unique(labeled_img))
  
    print('No. of GT Red Blobs: ', len(glst)-1)

    if mtd == 0:
        # Generate all separate CLEAR blobs:
        if mdl == 'Densenet':
            CLEAR_file = CLEAR_Densenet_path + 'CLEAR_VGG_point_heatmap_' + fname + '.csv'
        else:
            CLEAR_file = CLEAR_VGG_path + 'CLEAR_VGG_point_heatmap_' + fname + '.csv'
        clr = np.genfromtxt(CLEAR_file, delimiter=",")        #Change directory prefix to the heatmap location for clear


        clr_norm = (clr - clr.min())/(clr.max() - clr.min())

        pos = np.sort(np.unique(clr_norm))[::-1]
        pos = pos[pos>0.7]
    elif mtd == 1:
        if (len(os.listdir(org_diseased)) == 0) | (str('patient'+fname+'.jpg') not in os.listdir(org_diseased)):
            print('Please seek approval from CheXpert for original image. No suggested demo image found!')
        # Loading GRAD-CAM model
        if mdl == 'vgg':
            # VGG16-BN
            model = torchvision.models.vgg16_bn(pretrained=True)
            num_ftrs = model.classifier[6].in_features
            features = list(model.classifier.children())[:-1]
            features.extend([torch.nn.Linear(num_ftrs, 2)])
            model.classifier = torch.nn.Sequential(*features)
            sal_layer = 'features.43'
            mdl_wt = model_path +'VGG16_Neurips_CheXpert.pt'
        else:
            # DenseNet121
            model = torchvision.models.densenet121(pretrained=True)
            for param in model.parameters():
                param.requires_grad = True
            num_ftrs = model.classifier.in_features
            # print(num_ftrs)
            model.classifier = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 2)
                # torch.nn.Sigmoid()
            )
            sal_layer = 'features.norm5'
            mdl_wt = model_path + 'Densenet_Neurips_CheXpert.pt'
            print(mdl_wt)
        try:
            model.load_state_dict(torch.load(mdl_wt, map_location=torch.device('cpu'))['model_state_dict'].state_dict())
            print('Model weight loaded successfully!')
        except:
            print('Check path of pre-trained weights!')
        model = model.cuda()

        model.eval()
        input_image = Image.open(org_diseased+'/patient'+fname+'.jpg').convert('RGB')
        input_tensor = Preprocess_Xray(input_image)
        x = input_tensor.unsqueeze(0)
        x = x.to('cuda')
        GradCam_saliency = grad_cam(model, x.requires_grad_(True), 1, saliency_layer=sal_layer)  # vgg16 - features.43, densenet - features.norm5
        GradCam_saliency = GradCam_saliency[0][0].cpu().detach().numpy()
        gc_224 = cv2.resize(GradCam_saliency, (224,224))
        gc_224_norm = (gc_224 - gc_224.min())/(gc_224.max() - gc_224.min())
        gc_224_mask = np.where(gc_224_norm >=0.90, 1,0)
        gc_224_2nd = np.where((gc_224_norm < 0.90) & (gc_224_norm >=0.80), 1,0)
        gc_224_3rd = np.where((gc_224_norm < 0.80) & (gc_224_norm >=0.70), 1,0)

        gcmask = [gc_224_mask, gc_224_2nd, gc_224_3rd]
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
    cx = []
    cy = []
    pair = []
    totalgt+=1
    topj = np.min([len(pos),4])
    gtblob = np.where(labeled_img == 0, 0, 1)

    if mtd == 0:
        xaiblob = np.where(clr_norm >= pos[topj - 1], 1, 0)
    elif mtd == 1:
        gcm = np.zeros((224,224), dtype='uint8')
        for j in range(topj):
            gcm = cv2.bitwise_or(gcm, gcmask[j].astype('uint8'))
        xaiblob = gcm
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
if (mtd == 1):
    if (totalgt != 6):
        print('WARNING: Grad-CAM feature is limited without suggested diseased images from original source!')
# bstdf.to_csv('./ext_densenet_iou_xpert.csv') # Save specific IOU result as csv file