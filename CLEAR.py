""" This is the main module for CLEAR. CLEAR can either be run with:
(a) one of the sample models/datasets provided in CLEAR_sample_models_datasets.py .To do this run
    Run_CLEAR_with_sample_model()
(b) with a user created model and datasets. In this case run
     Run_CLEAR(X_train, X_test_sample, model, model_name, numeric_features, categorical_features, category_prefix, class_labels)
     An example of the required inputs is provided at the bottom of this module.
CLEAR's input parameters are specified in CLEAR_settings.py
 """

import numpy as np
import pandas as pd
from PIL import Image
import CLEAR_perturbations
import CLEAR_regression
import CLEAR_settings
import CLEAR_image
from datetime import datetime
import torch

def save_X_ray_image(filename, data):
    if type(data) == torch.Tensor:
        data = data.numpy()
    img = np.clip(data * 255, 0, 255)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img = img.resize((256, 256))
    if filename == 'Xray_CLEAR_diff.png':
        img = img.convert('1')
    img.save(CLEAR_settings.CLEAR_path+filename)

def Get_Xray_images(Xray):
    CLEAR_settings.init()
    org_file = CLEAR_settings.input_images_path + Xray
    org_image = Image.open(org_file).convert('RGB')
    org_image.save(CLEAR_settings.CLEAR_path+'Input_image.png')
    org_image.save('Input_image.png')

    cured_file= CLEAR_settings.GAN_images_path + 'GAN_' + Xray
    cured_image = Image.open(cured_file).convert('RGB')
    cured_image.save(CLEAR_settings.CLEAR_path+'GAN_image.png')
    return



def Run_CLEAR_Medical_Image():
    c_counter_master_df = pd.read_pickle(CLEAR_settings.CLEAR_path + "c_counter_master_df.pkl")
    results_master_df = pd.read_pickle(CLEAR_settings.CLEAR_path + "results_master_df.pkl")
    np.random.seed(1)
    Xrays_df = pd.read_csv(CLEAR_settings.CLEAR_path +  CLEAR_settings.images_to_process, index_col=None)
    cnt= 0
    while Xrays_df.loc[cnt,'Cured']==1:
        Xray = 'patient' + Xrays_df.loc[cnt,'id']
        CLEAR_settings.image_file_ID = Xray
        Get_Xray_images(Xray)
        org_img, model, top_label, top_idx = CLEAR_image.Create_PyTorch_Model()
        if CLEAR_settings.image_segment_type == 'felzenszwalb':
            max_num_Felz = 16
        else:
            max_num_Felz = 10
        img,segments,top_label,top_idx, model, target_num_other_large_seg= CLEAR_image.Segment_image(org_img,model,top_label,top_idx,max_num_Felz)
        explainer=CLEAR_image.Create_image_sensitivity(img,segments,top_label,top_idx, model)
        while explainer.counterfactuals is False and max_num_Felz > 7:
            max_num_Felz -=3
            if target_num_other_large_seg > 1:
                target_num_other_large_seg -=2
            img, segments, top_label, top_idx, model, target_num_other_large_seg= CLEAR_image.Segment_image(org_img, model, top_label,
                                                                top_idx,max_num_Felz,target_num_other_large_seg)
            explainer = CLEAR_image.Create_image_sensitivity(img, segments, top_label, top_idx, model)
        explainer, X_test_sample= CLEAR_image.Create_synthetic_images(explainer)
        explainer.batch = Xrays_df.loc[cnt,'id'][:-4]
        (results_df, explainer, single_regress) = CLEAR_regression.Run_Regressions(X_test_sample, explainer)
        (nncomp_df, missing_log_df) = CLEAR_perturbations.Calculate_Perturbations(explainer, results_df)
        c_counter_df=CLEAR_perturbations.Single_prediction_report(results_df, nncomp_df, single_regress, explainer)
        c_counter_master_df = c_counter_master_df.append(c_counter_df, ignore_index = True)
        results_master_df =  results_master_df.append(results_df, ignore_index=True)
        cnt += 1
    c_counter_master_df.to_pickle(
    CLEAR_settings.CLEAR_path + 'c_batch_df' + datetime.now().strftime("%Y%m%d-%H%M%S%f") + '.pkl')
    results_master_df.to_pickle(
    CLEAR_settings.CLEAR_path + 'results_batch_df' + datetime.now().strftime("%Y%m%d-%H%M%S%f") + '.pkl')
    c_counter_master_df.to_excel(CLEAR_settings.CLEAR_path +'Fidelity_Xrays.xlsx',index=False)
    results_master_df.to_excel(CLEAR_settings.CLEAR_path + 'Results_Xrays.xlsx', index=False)
    return



def Run_CLEAR_Synthetic_Data():
    c_counter_master_df = pd.read_pickle(CLEAR_settings.CLEAR_path + "c_counter_master_df.pkl")
    results_master_df = pd.read_pickle(CLEAR_settings.CLEAR_path + "results_master_df.pkl")
    np.random.seed(1)
    pointing_df = pd.read_csv(CLEAR_settings.CLEAR_path + CLEAR_settings.images_to_process, index_col=None, dtype = str)
    for pointing_index in range(0, min(100, pointing_df.shape[0])):
        file_num = pointing_df.loc[pointing_index, 'file']
        print(pointing_index)
        print(file_num)
        cured_file = CLEAR_settings.GAN_images_path + 'gen_healthy'+ file_num + '.png'
        cured_image = Image.open(cured_file).convert('RGB')
        cured_image.save(CLEAR_settings.CLEAR_path + 'GAN_image.png')
        org_file = CLEAR_settings.input_images_path +'diseased' + file_num + '.png'
        org_image = Image.open(org_file).convert('RGB')
        org_image.save(CLEAR_settings.CLEAR_path + 'Input_image.png')
        org_image.save('Input_image.png')
        np.random.seed(1)
        img, model, top_label, top_idx = CLEAR_image.Create_PyTorch_Model()
        if CLEAR_settings.image_segment_type == 'felzenszwalb':
            max_num_Felz = 16
        else:
            max_num_Felz = 10
        img,segments,top_label,top_idx, model, target_num_other_large_seg= CLEAR_image.Segment_image(img,model,top_label,top_idx,max_num_Felz)
        explainer = CLEAR_image.Create_image_sensitivity(img, segments, top_label, top_idx, model)
        explainer.batch = file_num
        explainer, X_test_sample = CLEAR_image.Create_synthetic_images(explainer)
        (results_df, explainer, single_regress) = CLEAR_regression.Run_Regressions(X_test_sample, explainer)
        if single_regress.perfect_separation is True:
            continue
        (nncomp_df, missing_log_df) = CLEAR_perturbations.Calculate_Perturbations(explainer, results_df)
        c_counter_df = CLEAR_perturbations.Single_prediction_report(results_df, nncomp_df, single_regress, explainer)
        c_counter_master_df = c_counter_master_df.append(c_counter_df, ignore_index = True)
        results_master_df =  results_master_df.append(results_df, ignore_index=True)
    c_counter_master_df.to_pickle(
    CLEAR_settings.CLEAR_path + 'c_batch_df' + datetime.now().strftime("%Y%m%d-%H%M%S%f") + '.pkl')
    results_master_df.to_pickle(
    CLEAR_settings.CLEAR_path + 'results_batch_df' + datetime.now().strftime("%Y%m%d-%H%M%S%f") + '.pkl')

    c_counter_master_df.to_excel(CLEAR_settings.CLEAR_path +'Fidelity_Xrays.xlsx',index=False)
    results_master_df.to_excel(CLEAR_settings.CLEAR_path + 'Results_Xrays.xlsx', index=False)
    return


if __name__ == "__main__":
    CLEAR_settings.init()
    if CLEAR_settings.case_study == 'Medical':
        Run_CLEAR_Medical_Image()
    elif CLEAR_settings.case_study == 'Synthetic':
         Run_CLEAR_Synthetic_Data()
    else:
        print('Case study mispecified')