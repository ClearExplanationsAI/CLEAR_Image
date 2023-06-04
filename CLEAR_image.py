import CLEAR_settings
import csv
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import copy
import pandas as pd
import CLEAR_regression
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from skimage import measure
from sklearn.cluster import DBSCAN, KMeans
from torchvision import transforms
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import (square, rectangle, diamond, disk, star)
from skimage.segmentation import felzenszwalb
from random import seed
from random import randint
from math import factorial


def Preprocess_Xray(input_image):
    if CLEAR_settings.image_model_name in ['VGG16_Synth_Classifier.pth','Dense_Synth_Classifier.pth']:
        preprocess = transforms.ToTensor()
    else:
        preprocess = transforms.Compose([
            transforms.Resize(230),
            transforms.CenterCrop(224),
            transforms.ToTensor()
            # transforms.GaussianBlur(5, sigma=(5, 10))
        ])
    input_tensor = preprocess(input_image)
    return(input_tensor)

def Create_PyTorch_Model():
    print(torch.__version__)
    model = torch.load(CLEAR_settings.CLEAR_path + CLEAR_settings.image_model_name)
    if torch.cuda.is_available():
        model.to('cuda')
    filename = CLEAR_settings.CLEAR_path + 'Input_image.png'
    input_image = Image.open(filename).convert('RGB')
    input_tensor = Preprocess_Xray(input_image)

    input_batch = input_tensor.unsqueeze(0)
    model.eval()
 # create a mini-batch as expected by the model
    preds = Get_image_predictions(model, input_batch)
    classes = CLEAR_settings.image_classes #change to get from model
    top_idx = 1  # for X-rays we have assigned pathology to class 1, needs to be improved to allow multiclass
    top_label = classes[top_idx]
    print(top_label, preds[0][1])
    return(input_batch, model,top_label, top_idx)


def Segment_image(img,model,top_label,top_idx, max_num_Felz=10, target_num_other_large_seg = 4):
    from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
    from skimage.segmentation import mark_boundaries
    from skimage.util import img_as_float
    img =img.permute(0, 2, 3,1 ).cpu().numpy()
    if CLEAR_settings.image_segment_type == 'quickshift':
        segments = quickshift((img_as_float(img[0]).astype(np.float64)), kernel_size=20,  # kernel set to 10 for synthetic data
                                                max_dist=10, ratio=2)
        if CLEAR_settings.debug_mode is True:
            plt.imshow(segments)
            plt.show()


    elif CLEAR_settings.image_segment_type == 'felzenszwalb':
        Felz_min_size = 250
        successful_Felz = False
        while successful_Felz is False:
            segments = felzenszwalb((img_as_float(img[0]).astype(np.float64)), scale=25, sigma=0, min_size=Felz_min_size)
            # plt.imshow(segments)
            # plt.show()
            if len(np.unique(segments)) > 16:  # 'in both' was 25
                Felz_min_size += 25
            else:
                successful_Felz = True
        print(f"Felzenszwalb number of segments: {len(np.unique(segments))}")
        if CLEAR_settings.debug_mode is True:
            plt.imshow(segments)
            plt.show()


    elif CLEAR_settings.image_segment_type == 'GAN_Xrays':
        segments = Xray_seg(img, max_num_Felz ,target_num_other_large_seg).astype(np.uint8)
    elif CLEAR_settings.image_segment_type == 'GAN_diff':
        segments = GAN_seg(img).astype(np.uint8)

    else:
        print('input error for segmentation choice')
        exit()
    if CLEAR_settings.debug_mode is True:
        plt.imshow(mark_boundaries(img[0], segments))
        plt.tight_layout()
        plt.show()
    return(img,segments,top_label,top_idx, model,target_num_other_large_seg)


def Create_image_sensitivity(img,segments,top_label,top_idx, model):
    #NB This has been coded on the assumption that Seg 0 is background and should not be a component of a counterfactual

    def get_predictions(data, segments, GAN_array, predictions, model):
        imgs = []
        for row in tqdm(data):
            infilled_img = Create_infill(row, segments, img, GAN_array)
            infilled_img = torch.from_numpy(infilled_img).permute(2, 0, 1)
            imgs.append(infilled_img)
            if len(imgs) == batch_size:
                preds = Get_image_predictions(model, imgs)
                predictions.extend(preds)
                imgs = []
        if len(imgs) > 0:
            preds = Get_image_predictions(model, imgs)
            predictions.extend(preds)
        return (predictions)

    GAN_array = None
    batch_size = 64
    if CLEAR_settings.image_infill =='GAN':
        GAN_image = Image.open(CLEAR_settings.CLEAR_path +'GAN_image.png').convert('RGB')
        GAN_image= Preprocess_Xray(GAN_image)
        input_batch = GAN_image.unsqueeze(0)
        GAN_pred = Get_image_predictions(model, input_batch)[0][top_idx]
        GAN_array = GAN_image.permute(1,2,0).numpy()

    if CLEAR_settings.image_segment_type == 'feature_map':
        unique_segments = [x for x in range(0,segments.max()+1)]
    else:
        unique_segments = np.unique(segments)


    if CLEAR_settings.image_use_old_synthetic is False:
        print('\n Performing grid search - step 1 of CLEAR method \n')
        try:
            os.remove(CLEAR_settings.CLEAR_path + 'catSensitivity.csv')
        except OSError:
            pass
    #create feature data for sensitivity files
        top_row = ['observation', 'feature1', 'feature2', 'feature3','feature4', 'probability']
        features = [str(x) for x in range(np.unique(segments).shape[0])]
        features = ["S0" + x if len(x) == 1 else "S" + x if len(x) == 2 else x for x in
                                features]
        features = [str(x) + "DdSeg" + str(x[1:]) for x in features]

        seg_pair_1 = []
        seg_pair_2 = []
        seg_pair_3 = []
        seg_pair_4 = []
        num_features = len(unique_segments)
        #number_combs excludes 0 which is background
        num_features_excl_zero = num_features -1
        #number of combinations for 2-deep counterfactuals
        if num_features_excl_zero >= 2:
            number_combs = (num_features_excl_zero+ int(factorial(num_features_excl_zero) / (2* factorial(num_features_excl_zero - 2))))
        else:
            number_combs = num_features_excl_zero
        data = np.ones((number_combs, num_features))
        unique_segments_excl_zero = unique_segments.tolist()
        unique_segments_excl_zero.remove(0)
        features.remove('S00DdSeg00')
        cnt = 0
        for j in unique_segments_excl_zero:
            data[cnt,j] = 0
            seg_pair_1.append(features[cnt])
            seg_pair_2.append('nil')
            seg_pair_3.append('nil')
            seg_pair_4.append('nil')
            cnt +=1
        if num_features_excl_zero >= 2:
            other_segments = copy.deepcopy(unique_segments_excl_zero)
            for j in unique_segments_excl_zero:
                other_segments.remove(j)
                for i in other_segments:
                    data[cnt,j] = data[cnt,i] = 0
                    seg_pair_1.append(features[j-1])  #-1 because 'S00DdSeg00' removed
                    seg_pair_2.append(features[i-1])
                    seg_pair_3.append('nil')
                    seg_pair_4.append('nil')
                    cnt +=1

        predictions = []
        predictions = get_predictions(data, segments, GAN_array, predictions, model)
        predictions_for_top_class = [x[top_idx] for x in predictions]
        idx = [index for index, value in enumerate(predictions_for_top_class) if value < 0.5]
        if len(idx) == 0 and num_features_excl_zero > 2:
            number_combs = int(factorial(num_features_excl_zero) / (factorial(3) * factorial(num_features_excl_zero - 3)))
            data = np.ones((number_combs, num_features))
            other_segments = copy.deepcopy(unique_segments_excl_zero)
            cnt=0
            for j in unique_segments_excl_zero:
                other_segments.remove(j)
                other_segments_2 = copy.deepcopy(other_segments)
                for i in other_segments:
                    if i in other_segments_2:
                        other_segments_2.remove(i)
                    else:
                        continue
                    for k in other_segments_2:
                        data[cnt,j] = data[cnt,i] = data[cnt,k] = 0
                        seg_pair_1.append(features[j-1])
                        seg_pair_2.append(features[i-1])
                        seg_pair_3.append(features[k-1])
                        seg_pair_4.append('nil')
                        cnt +=1
            predictions = get_predictions(data, segments, GAN_array, predictions, model)


        add_4_deep_if_needed = True
        predictions_for_top_class = [x[top_idx] for x in predictions]
        idx = [index for index, value in enumerate(predictions_for_top_class) if value < 0.5]
        if len(idx) == 0 and add_4_deep_if_needed is True and num_features_excl_zero > 4:
            print('searching for 4-deep counterfactuals')
            cnt = 0
            # number of additional combinations for 4-deep counterfactuals
            number_combs = int(factorial(num_features_excl_zero) / (factorial(4) * factorial(num_features_excl_zero - 4)))
            data = np.ones((number_combs, num_features))
            other_segments = copy.deepcopy(unique_segments_excl_zero)
            for j in unique_segments_excl_zero:
                other_segments.remove(j)
                other_segments_2 = copy.deepcopy(other_segments)
                for i in other_segments:
                    other_segments_2.remove(i)
                    other_segments_3 =copy.deepcopy(other_segments_2)
                    for k in other_segments_2:
                        other_segments_3.remove(k)

                        for m in other_segments_3:
                            data[cnt,j] = data[cnt,i] = data[cnt,k] = data[cnt,m]=0
                            seg_pair_1.append(features[j-1])
                            seg_pair_2.append(features[i-1])
                            seg_pair_3.append(features[k-1])
                            seg_pair_4.append((features[m-1]))
                            cnt +=1
            predictions = get_predictions(data, segments, GAN_array, predictions, model)
        cnt = 0
        with open(CLEAR_settings.CLEAR_path + 'catSensitivity.csv', 'w', newline='') as file1:
            writes = csv.writer(file1, delimiter=',', skipinitialspace=True)
            writes.writerow(top_row)
            try:
                while cnt < len(predictions):
                    observation = 0
                    forecast=predictions[cnt][top_idx]
                    segment_pair_1 = seg_pair_1[cnt]
                    segment_pair_2 = seg_pair_2[cnt]
                    segment_pair_3 = seg_pair_3[cnt]
                    segment_pair_4 = seg_pair_4[cnt]
                    out_list = [observation, segment_pair_1,segment_pair_2,segment_pair_3,segment_pair_4,forecast]
                    writes.writerow(out_list)
                    cnt += 1
            except:
                pass
        file1.close()

    numeric_features=[]
    temp= "not "+ top_label
    class_labels= [temp,top_label]
    unique_segments= np.unique(segments)
    if CLEAR_settings.image_segment_type == 'feature_map':
        unique_segments = [x for x in range(0,segments.max()+1)]
    categorical_features = [str(x) for x in range(len(unique_segments))]
    categorical_features = ["S0" + x if len(x) == 1 else "S" + x if len(x) == 2 else x for x in categorical_features]
    categorical_features= [str(x) +"DdSeg"+str(x[1:]) for x in categorical_features]
    category_prefix = [x[0:3] for x in categorical_features]

    explainer = CLEAR_regression.Create_explainer(model, class_labels,categorical_features,
                                                  category_prefix, numeric_features, data_type='image')
    explainer.cat_features= categorical_features
    explainer.feature_list = categorical_features
    explainer.num_features =len(unique_segments)
    explainer.numeric_features = []
    explainer.image = img
    explainer.top_idx = top_idx
    explainer.segments= segments
    explainer.GAN_array = GAN_array  # this defaults to  None
    explainer.batch = 'None'
    explainer.counterfactuals = False
    if CLEAR_settings.image_infill =='GAN':
        explainer.GAN_image = GAN_image
        explainer.GAN_pred = GAN_pred

    y = Get_image_predictions(model, img)
    y = y[0][top_idx]
    temp_df = explainer.catSensit_df[
    (explainer.catSensit_df['observation'] == 0) &
    ((explainer.catSensit_df.probability >= CLEAR_settings.binary_decision_boundary)
     != (y >= CLEAR_settings.binary_decision_boundary))
    & (explainer.catSensit_df['feature1'] != 'S00DdSeg00') ].copy(deep=True)
    if temp_df.empty:
        print('grid search found no counterfactuals')
    else:
        explainer.counterfactuals = True
    return(explainer)

def Create_synthetic_images(explainer):
    batch_size = 64
    zeros_in_row = 5
    zero_image = explainer.image[0].copy()
    zero_image[:] = 0
    data = np.ones((CLEAR_settings.num_samples, explainer.num_features))
    for row in data:
        num_changed = np.random.randint(1, zeros_in_row)
        to_zero= np.random.randint(0, explainer.num_features, num_changed)  # WAS 3 or 6
        row[to_zero]= 0
    predictions = []
    data[0, :] = 1
    imgs = []
    if CLEAR_settings.image_use_old_synthetic is False:
#        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        for row in tqdm(data):
            infilled_img = Create_infill(row, explainer.segments, explainer.image, explainer.GAN_array)
            # plt.imshow(infilled_img/ 2 + 0.5)
            # plt.show()
            infilled_img = torch.from_numpy(infilled_img).permute(2, 0, 1)
            imgs.append(infilled_img)
            if len(imgs) == batch_size:
                preds= Get_image_predictions(explainer.model, imgs)
                predictions.extend(preds)
                imgs = []
        if len(imgs) > 0:
            preds= Get_image_predictions(explainer.model, imgs)
            predictions.extend(preds)
        explainer.master_df = pd.DataFrame(data=data, columns=explainer.feature_list)
        explainer.master_df.add_prefix('ImgDd')
        explainer.master_df['prediction']=[el[explainer.top_idx] for el in predictions]
        explainer= Check_synthetic_data(explainer, zeros_in_row, batch_size)
        X_test_sample = pd.DataFrame(columns=explainer.feature_list)
        X_test_sample= X_test_sample.append(explainer.master_df.iloc[0,:-1])
        explainer.master_df.to_pickle('master_df.pkl')
        X_test_sample.to_pickle('X_test_sample.pkl')
    else:
        explainer.master_df = pd.read_pickle('master_df.pkl')
        X_test_sample = pd.read_pickle('X_test_sample.pkl')

    return explainer, X_test_sample


def Check_synthetic_data(explainer,zeros_in_row,batch_size):
    explainer.poor_data = False
    key_data_df = explainer.master_df[explainer.master_df.prediction.between(0.02,.98)]
    if key_data_df.shape[0] == 0:
        print("No synthetic data in range 0.02 to 0.98")
        explainer.poor_data = True
    elif key_data_df.shape[0] < CLEAR_settings.num_samples/5:
        required_rows = CLEAR_settings.num_samples/5 - key_data_df.shape[0]
        key_data_df.drop(['prediction'], axis=1)
        new_data_df = pd.DataFrame(columns=key_data_df.columns)
        cnt = 0
        created_required_rows = False
        while created_required_rows is False:
            while new_data_df.shape[0] < required_rows * 1.2:
                i = randint(0,key_data_df.shape[0]-1)
                temp = key_data_df.iloc[i,:].copy(deep=True)
                col = np.random.randint(0, explainer.num_features, 3)
                temp.iloc[col]= abs(temp.iloc[col[0]]-1)
                new_data_df = new_data_df.append(temp, ignore_index=True)
                temp2 = key_data_df.iloc[i, :].copy(deep=True)
                temp2.iloc[col[1]] = abs(temp.iloc[col[1]] - 1)
                temp2.iloc[col[2]] = abs(temp.iloc[col[2]] - 1)
                new_data_df = new_data_df.append(temp, ignore_index=True)
                cnt +=1
                if cnt > CLEAR_settings.num_samples * 2:
                    print('failed to generate enough synthetic data')
                    created_required_rows = True
                    continue
            if new_data_df.shape[0]>CLEAR_settings.num_samples:
                new_data_df = new_data_df.sample(n=CLEAR_settings.num_samples)

            predictions = []
            imgs = []
            print('extra synthetic data being added')
            for i in tqdm(range(new_data_df.shape[0])):
                infilled_img = Create_infill(new_data_df.iloc[i,:], explainer.segments, explainer.image, explainer.GAN_array)
                # plt.imshow(infilled_img/ 2 + 0.5)
                # plt.show()
                infilled_img = torch.from_numpy(infilled_img).permute(2, 0, 1)
                imgs.append(infilled_img)
                if len(imgs) == batch_size:
                    preds = Get_image_predictions(explainer.model, imgs)
                    predictions.extend(preds)
                    imgs = []
            if len(imgs) > 0:
                preds = Get_image_predictions(explainer.model, imgs)
                predictions.extend(preds)
            new_data_df['prediction'] = [el[explainer.top_idx] for el in predictions]
            new_data_df = new_data_df[new_data_df.prediction.between(0.02,.98)]
            if new_data_df.shape[0] >= required_rows:
                created_required_rows = True
                explainer.master_df = explainer.master_df.append(new_data_df, ignore_index=True)
    return(explainer)

def Create_infill(row, segments, img, GAN_array):
        zero_image = img[0].copy()
        zero_image[:] = 0
        zeros = np.where(row == 0)[0]
        mask = np.zeros(segments.shape).astype(bool)
        for z in zeros:
            mask[segments == z] = True
        # Infilling with Opencv requires source image to be uint8. Therefore undo the preprocess transformation
        # performed by Keras.
        if CLEAR_settings.image_infill == 'telea':
            #openCV_img = ((img[0] + 1) * 127.5).astype(np.uint8)
            # convert to range [0,1] (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
            old_min= img[0].min()
            old_max= img[0].max()
            new_min= 0
            new_max=255
            openCV_img= (((img[0]- old_min)*(new_max-new_min))/(old_max-old_min) + new_min).astype(np.uint8)
            mask = mask.astype(np.uint8)
            infilled_img = cv2.inpaint(openCV_img, mask, 3, cv2.INPAINT_TELEA)
        # convert back to preprocessing format
            infilled_img = ((infilled_img - new_min) * (old_max - old_min)/(new_max-new_min))+ old_min
            #infilled_img = (infilled_img / 127.5 - 1).astype(np.float32)
        elif CLEAR_settings.image_infill == 'average':
            infilled_img = copy.deepcopy(img[0])
            for k in range(img[0].shape[2]):
                infilled_img[:, :, k][mask] = cv2.mean(img[0])[k]
                # plt.imshow(infilled_img / 2 + 0.5)
                # plt.show()
        elif CLEAR_settings.image_infill == 'black':
            infilled_img = copy.deepcopy(img[0])
            for k in range(img[0].shape[2]):
                infilled_img[:, :, k][mask] = 0
        elif CLEAR_settings.image_infill == 'GAN':
            infilled_img = copy.deepcopy(img[0])
            infilled_img[mask] = GAN_array[mask]
            if CLEAR_settings.debug_mode is True:
                plt.imshow(infilled_img)
                plt.show()
            # save_image('connected_seg.png', infilled_img)


        elif CLEAR_settings.image_infill == 'none':
            infilled_img = copy.deepcopy(img[0])
            infilled_img[mask] = zero_image[mask]
        else:
            print('error with inpainting input parameter')
            exit()
        return infilled_img

def save_image(filename, data):
    if type(data)==torch.Tensor:
        data = data.numpy()
    img = np.clip(data*255 ,0,255)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save(filename)
    return()


def Get_image_predictions(model,x):
    if isinstance(x, np.ndarray) and len(x.shape)==4:
        batch = torch.from_numpy(x).permute(0, 3, 1, 2)
    elif isinstance(x, np.ndarray) and len(x.shape)==3:
        batch = torch.from_numpy(x).permute(2, 0, 1)
    elif isinstance(x,list):
        batch = torch.stack(x, dim=0)
    else:
        batch = x
    if torch.cuda.is_available():
        batch = batch.to('cuda')
    with torch.no_grad():
        preds = model(batch)
    preds = F.softmax(preds, dim=1)
    preds = preds.cpu().detach().numpy()
    return(preds)


def Xray_seg(org_array, max_num_Felz, target_num_other_large_seg):
    #need to replace this as opening same file twice
    min_large_blob_size = 250
    erosion_disk_size = 2.2
    Felz_min_size = 250
    GAN_image = Image.open(CLEAR_settings.CLEAR_path +'GAN_image.png').convert('L')
    GAN_image = Preprocess_Xray(GAN_image)
    GAN_array = GAN_image[0].numpy()
    num_high_threshold_blobs = 0
    threshold_cnt =0
    high_threshold = 0.3
    while num_high_threshold_blobs == 0 and threshold_cnt < 5:
        high_threshold_image = abs(org_array[0,:,:,0] - GAN_array)
        high_threshold_map = (high_threshold_image > high_threshold).astype(np.uint8)
        high_threshold_blobs = measure.label(high_threshold_map,connectivity=1)
        unique, counts = np.unique(high_threshold_blobs, return_counts=True)
        high_threshold_blob_list = [i for (i, j) in zip(unique, counts) if (j >= min_large_blob_size and i != 0)]
        num_high_threshold_blobs = len(high_threshold_blob_list)
        threshold_cnt +=1
        high_threshold -= 0.05
    # 0 excluded as corresponds to background

    high_threshold_blob_bool = np.isin(high_threshold_blobs, high_threshold_blob_list)



    if CLEAR_settings.debug_mode is True:
        plt.imshow(high_threshold_blob_bool)
        plt.show()
    low_thres_diff_image = abs(org_array[0, :, :, 0] - GAN_array)
    low_thres_diff_map = (low_thres_diff_image > 0.035).astype(np.uint8)  # was 0.005 used 0.2 with sythetic check
    if CLEAR_settings.debug_mode is True:
        plt.imshow(low_thres_diff_map)
        plt.show()
    low_thres_diff_map[high_threshold_blob_bool==1]=0
    # plt.imshow(low_thres_diff_map)
    # plt.show()
        
    all_blobs = measure.label(low_thres_diff_map,connectivity=1)
    unique, counts = np.unique(all_blobs, return_counts=True)
    # 0 excluded as corresponds to background
    num_blobs= len(counts)
    if num_blobs == 1:
        print('GAN Xray is blank')
        quit()
    large_blob_list = [i for (i, j) in zip(unique, counts) if (j >= min_large_blob_size and i != 0)]
    large_blob_bool = np.isin(all_blobs, large_blob_list)
    #Erodes the large blobs setting pixels to zero, and updates the diff_map (diff map pixels = 1 being the pixels
    #that will be assigned to segments
    if CLEAR_settings.debug_mode is True:
        plt.imshow(large_blob_bool)
        plt.show()

    apply_erosion = True
    if apply_erosion is True:
        successful_erosion = False
        erosion_cnt = 0
        while not successful_erosion:
            selem = disk(erosion_disk_size)
            after_erosion_large_blob_bool = erosion(large_blob_bool, selem)
            if after_erosion_large_blob_bool.sum() / large_blob_bool.sum() < 0.70:
                erosion_disk_size = erosion_disk_size * 0.9
                erosion_cnt +=1
                if erosion_cnt == 20:
                    continue
            else:
                successful_erosion = True
                #transfer small eroded blobs to eroded array
                eroded_size_check = measure.label(after_erosion_large_blob_bool,connectivity=1)
                unique, counts = np.unique(eroded_size_check, return_counts=True)
                small_size_eroded_list = [i for (i, j) in zip(unique, counts) if (j < min_large_blob_size*.8 and i != 0)]
                mask = np.zeros(after_erosion_large_blob_bool.shape).astype(bool)
                for x in small_size_eroded_list:
                    mask[eroded_size_check==x]= True
                after_erosion_large_blob_bool[mask]= False

                eroded_array = np.zeros(all_blobs.shape).astype(np.int64)
                eroded_array[np.logical_and(np.invert(after_erosion_large_blob_bool), large_blob_bool)] = 1
                if CLEAR_settings.debug_mode is True:
                    plt.imshow(after_erosion_large_blob_bool)
                    plt.show()
                low_thres_diff_map[large_blob_bool]=0
                low_thres_diff_map[after_erosion_large_blob_bool] = 1
                #need to reassign blob labels as erosion may have changed number of blobs (eg split large blob in 2)
                all_blobs_after_erosion = measure.label(low_thres_diff_map,connectivity=1)
                unique, counts = np.unique(all_blobs_after_erosion, return_counts=True)
                large_blob_list = [i for (i, j) in zip(unique, counts) if (j >= min_large_blob_size*.8 and i != 0)] # lower number as eroded pixels
                                                                                                # will be restored. This means that
                large_blob_bool = np.isin(all_blobs_after_erosion, large_blob_list)             #there will be some blobs sizes
                                                                                        #that are uneroded.
    large_blob_array = np.zeros(all_blobs_after_erosion.shape).astype(np.int16)
    large_blob_array[large_blob_bool]= all_blobs_after_erosion[large_blob_bool]
    #add back the eroded away pixels
    if apply_erosion is True:
        mask = np.zeros(eroded_array.shape).astype(bool)
        k = 0
        while k <100 and eroded_array[eroded_array==1].sum() != 0:
            k +=1
            r, c = np.where(eroded_array == 1)
            for m in range(0, r.shape[0]):
                if any([r[m]==0,c[m]==0,r[m]==eroded_array.shape[0] -1, c[m]==eroded_array.shape[0] - 1])== True:
                    continue
                max_neighbour = max([large_blob_array[r[m], c[m]],large_blob_array[r[m] - 1, c[m] - 1],
                                     large_blob_array[r[m],c[m] - 1],large_blob_array[r[m] + 1][c[m] - 1],
                                     large_blob_array[r[m] - 1][c[m]],large_blob_array[r[m] + 1][c[m]],
                                     large_blob_array[r[m] - 1][c[m] + 1],large_blob_array[r[m]][c[m] + 1],
                                     large_blob_array[r[m] + 1][c[m] + 1]])
                if max_neighbour != 0:
                    eroded_array[r[m],c[m]]= max_neighbour
                    mask[r[m],c[m]] = True
            large_blob_array[mask]= eroded_array[mask]
            if CLEAR_settings.debug_mode is True:
                plt.imshow(large_blob_array)
                plt.show()
    # keep any blobs that were eroded away and remain unassigned and size > 30 pixels
    unassigned_eroded = np.zeros(eroded_array.shape).astype(bool)
    unassigned_eroded[eroded_array==1] = 1
    unassigned_eroded= measure.label(unassigned_eroded, connectivity=1)
    unique, counts = np.unique(unassigned_eroded, return_counts=True)
    unassigned_eroded_list = [i for (i, j) in zip(unique, counts) if (j >= 30 and i != 0)]


    # Creates a temp boolean for plotting
    if CLEAR_settings.debug_mode is True:
        temp = (large_blob_array > 0.005).astype(np.bool)
        temp = np.clip(temp * 255, 0, 255)
        plt.imshow(temp)
        plt.title('large blobs')
        plt.show()

    #segement largest blobs using Felzenszwalb on GAN image
    unique, counts = np.unique(large_blob_array, return_counts=True)
    largest_blobs = unique[counts>1000].tolist()
    try:
        largest_blobs.remove(0)
    except:
        pass
    largest_blobs_mask = np.isin(large_blob_array,largest_blobs)


    successful_Felz = False
    while successful_Felz is False:
        Felz_array = np.zeros(largest_blobs_mask.shape)
        Felz_array[largest_blobs_mask] = GAN_array[largest_blobs_mask]
        Felz_array = felzenszwalb(Felz_array, scale=25, sigma=0, min_size=Felz_min_size)
        Felz_array[largest_blobs_mask== False] = 0
        # plt.imshow(Felz_array)
        # plt.show()
        if len(np.unique(Felz_array)) > max_num_Felz: #was 25
            Felz_min_size += 25
        else:
            successful_Felz = True
    print(f"Felzenszwalb number of segments: {len(np.unique(Felz_array))}")
    # The felzenszwalb algorithm is expanding the segements, which needs to be corrected
    Felz_array[largest_blobs_mask == False] = 0
    if CLEAR_settings.debug_mode is True:
        plt.imshow(Felz_array)
        plt.show()
    for blob in high_threshold_blob_list:
        Felz_array[high_threshold_blobs == blob] = len(np.unique(Felz_array)) + 100

    Felz_blobs = np.unique(Felz_array).tolist()
    cnt= 0
    Sequential_Felz_array = np.zeros(Felz_array.shape).astype(np.uint64)
    for i in Felz_blobs:
        Sequential_Felz_array[Felz_array==i]=cnt
        cnt +=1


    other_large_size = 200  # keep number of additional segments to less than 5
    num_other_large_seg = 999
    while num_other_large_seg > target_num_other_large_seg and other_large_size < 1000 :
        other_large_blobs= unique[np.where((counts < 1000) & (counts > other_large_size))].tolist()
        num_other_large_seg= len(other_large_blobs)
        other_large_size += 100
    print('small blobs ' + str(num_other_large_seg))

    smallest_large_bool = np.isin(large_blob_array, other_large_blobs)
    smallest_large_array= np.zeros(smallest_large_bool.shape).astype(np.uint16)
    #smallest_large_array[smallest_large_bool] = large_blob_array[smallest_large_bool]
    cnt = len(np.unique(Sequential_Felz_array))
    for i in other_large_blobs:
        smallest_large_array[large_blob_array==i]=cnt
        cnt +=1
    large_blob_array =smallest_large_array + Sequential_Felz_array
    large_blob_bool= (large_blob_array > 0.005).astype(np.bool)
    unique, counts = np.unique(large_blob_array, return_counts=True)
    num_large_blobs = len(counts)
    segments = large_blob_array.astype(int)
    include_small_blobs = False
    if include_small_blobs:
        small_blob_array = np.zeros(all_blobs.shape).astype(np.int64)
        small_blob_array[np.invert(large_blob_bool)] = all_blobs[np.invert(large_blob_bool)]
        small_blob_bool = (small_blob_array > 0.005).astype(np.bool)
    # Creates a temp boolean for plotting
        if small_blob_bool.sum() > 0:
            if CLEAR_settings.debug_mode is True:
                temp = np.clip(small_blob_bool * 255, 0, 255)
                if CLEAR_settings.debug_mode is True:
                    plt.title('small blobs')
                    plt.imshow(temp)
                    plt.show()
            small_cluster = np.zeros((all_blobs.shape[0] * small_blob_bool.shape[0], 2))
            small_blob_pixels_cnt = 0
            for i in np.arange(small_blob_bool.shape[0]):
                for j in np.arange(small_blob_bool.shape[1]):
                    if small_blob_bool[i, j] == True:
                        small_cluster[small_blob_pixels_cnt, 0] = i
                        small_cluster[small_blob_pixels_cnt, 1] = j
                        small_blob_pixels_cnt += 1
            to_cluster = small_cluster[0:small_blob_pixels_cnt, :]
            kmeans = KMeans(n_clusters=4, random_state=0).fit(to_cluster)
            labels = kmeans.labels_
            labels += num_large_blobs
            small_blob_array = np.zeros((all_blobs.shape[0], all_blobs.shape[1]))
            for i in np.arange(small_blob_pixels_cnt):
                small_blob_array[np.int(to_cluster[i][0]), np.int(to_cluster[i][1])] = labels[i]
            if CLEAR_settings.debug_mode is True:
                plt.imshow(small_blob_array)
                plt.show()
            segments += small_blob_array.astype(int)
            # if CLEAR_settings.debug_mode is True:
            if CLEAR_settings.debug_mode is True:
                plt.imshow(segments)
                plt.show()
    print('number segments: ' + str(np.int(segments.max())))
    return (segments)


def GAN_seg(org_array):
    GAN_image = Image.open(CLEAR_settings.CLEAR_path +'GAN_image.png').convert('L')
    GAN_image = Preprocess_Xray(GAN_image)
    GAN_array = GAN_image[0].numpy()
    high_threshold = 0.25  #was 0.1 for simple case study then 0. then 0.25
    high_threshold_image = abs(org_array[0, :, :, 0] - GAN_array)
    high_threshold_map = (high_threshold_image > high_threshold).astype(np.uint8)
    selem = disk(1)
    all_segments = measure.label(dilation(high_threshold_map,selem), connectivity=1)
    # all_segments = measure.label(high_threshold_map,connectivity=1)
    if CLEAR_settings.debug_mode == True:
        plt.imshow(all_segments)
        plt.show()
    unique, counts = np.unique(all_segments, return_counts=True)

    large_blob_list = [i for (i, j) in zip(unique, counts) if (j >= 50 and i != 0)] #was 50
    large_blob_bool = np.isin(all_segments, large_blob_list)
    segments = np.zeros(large_blob_bool.shape).astype(np.int64)
    segments[large_blob_bool]= all_segments[large_blob_bool]
    segments = measure.label(segments, connectivity=1)
    unique, counts = np.unique(segments, return_counts=True)
    if CLEAR_settings.debug_mode is True:
        plt.imshow(segments)
        plt.show()

    print('number segments: ' + str(len(counts)))
    return (segments)