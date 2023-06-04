import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom
import pandas as pd
import torchray.attribution.extremal_perturbation as elp
from torchray.attribution.grad_cam import grad_cam
from skimage.segmentation import mark_boundaries
import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("output_noGreen.pdf")
from captum.attr import  LayerGradCam
import matplotlib.colors as mcolors
import cv2
import io

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
model_type = 'Densenet'



pdf = matplotlib.backends.backend_pdf.PdfPages("Pointing_Game_outputs.pdf")
def Get_Grid_Max(saliency_image):
    Num_pix = image_size / grid_dim
    square_grid = np.zeros((grid_dim,grid_dim), dtype= np.float16)
    for i in range(grid_dim):
        for j in range(grid_dim):
            start_i = i * round(Num_pix)
            end_i = start_i + round(Num_pix)
            if end_i > image_size - Num_pix:
                end_i = image_size
            start_j = j * round(Num_pix)
            end_j = start_j + round(Num_pix)
            if end_j > image_size - Num_pix:
                end_j = image_size
            square_grid[i,j] = np.average(saliency_image[start_i:end_i, start_j:end_j])
    return(square_grid)


def Create_square_grid():
    Num_pix = image_size / grid_dim
    square_grid = np.zeros((image_size,image_size), dtype= np.int)
    for i in range(grid_dim):
        for j in range(grid_dim):
            start_i = i * round(Num_pix)
            end_i = start_i + round(Num_pix)
            if end_i > image_size - Num_pix:
                end_i = image_size
            start_j = j * round(Num_pix)
            end_j = start_j + round(Num_pix)
            if end_j > image_size - Num_pix:
                end_j = image_size
            square_grid[start_i:end_i, start_j:end_j] = i * grid_dim + j
    square_grid = square_grid.astype(np.int)
    return(square_grid)

def Largest_indices(scores):
    flat = scores.flatten()
    indices = np.arange(0, len(flat))
    indices = indices[np.argsort(-flat[indices])]
    same_value = np.zeros(indices.shape)
    same_value[0]=1
    cnt = 1
    for i in range(0,same_value.shape[0]-1):
        if flat[indices[i]]== flat[indices [i+1]]:
            same_value[i+1] = same_value[i]
        else:
            cnt +=1
            same_value[i+1] = cnt
    return indices, same_value

def Largest_Area_indices(scores):
    flat = scores.flatten()


    indices = np.arange(0, len(flat))
    indices = indices[np.argsort(-flat[indices])]
    same_value = np.zeros(indices.shape)
    same_value[0]=1
    cnt = 1
    for i in range(0,same_value.shape[0]-1):
        if flat[indices[i]]== flat[indices [i+1]]:
            same_value[i+1] = same_value[i]
        else:
            cnt +=1
            same_value[i+1] = cnt
    return indices, same_value

def Point_test_Equal(technique, score_array):
    #set any score less than 1% of max to zero.
    # This is prevent 'invisible' scores influencing the pointing game
    max_score = score_array.max()
    score_array[abs(score_array) < max_score/100]= 0
    top_scores, same_scores = Largest_indices(score_array)
    unique_scores = np.unique(same_scores).astype(np.int)
    relevant_features= pointing_df.loc[pointing_index,'relevant_features']
    num_relevant_features = len(relevant_features)
    relevant_found = np.zeros(num_relevant_features, dtype = int)
    hits = misses = 0
    point_success = False
    score_index = 0
    for i in unique_scores:
        if point_success is True:
            break
        while i == same_scores[score_index]:
            irrelevant = True       # CLEAR score has not been matched
            feature_cnt = 0
            for j in relevant_features:
                if j == 'not_square_inside':
                    rel_column = 'square_inside'
                else:
                    rel_column = j
                if np.isin(top_scores[score_index], pointing_df.loc[pointing_index, rel_column]):
                    relevant_found[feature_cnt] = 1
                    hits += 1
                    irrelevant = False
                feature_cnt += 1
            if irrelevant == True and np.isin(0, relevant_found):
                misses += 1
            elif score_index != same_scores.shape[0]-1:
                if irrelevant == True and same_scores[score_index] == same_scores[score_index + 1]:
                    misses += 1
            elif irrelevant == True and score_index == same_scores.shape:
                misses += 1
            elif not np.isin(0, relevant_found):
                point_success = True
            score_index += 1
            if score_index == 48:
                pass
            if score_index == same_scores.shape[0]:
                break
    point_score = hits/(hits + misses)
    return(point_success, point_score, relevant_features)

def Point_test(technique, score_array):
    #set any score less than 1% of max to zero.
    # This is prevent 'invisible' scores influencing the pointing game
    max_score = score_array.max()
    score_array[abs(score_array) < max_score/100]= 0
    top_scores, same_scores = Largest_indices(score_array)
    relevant_features= pointing_df.loc[pointing_index,'relevant_features']
    relevant_features = [x for x in relevant_features if not x.startswith('G')]
    num_relevant_features = len(relevant_features)
    relevant_found = np.zeros(num_relevant_features, dtype = int)
    hits = misses = 0
    point_success = False
    for i in top_scores:
        irrelevant = True
        feature_cnt = 0
        for j in relevant_features:
            if j == 'not_square_inside':
                rel_column = 'square_inside'
            else:
                rel_column = j
            if np.isin(i, pointing_df.loc[pointing_index,rel_column]):
                relevant_found[feature_cnt]=1
                hits += 1
                irrelevant = False
            feature_cnt += 1
        if irrelevant == True and np.isin(0, relevant_found):
            misses += 1
        elif not np.isin(0, relevant_found):
            point_success = True
            break
    point_score = hits/(hits + misses)
    return(point_success, point_score, relevant_features)



def Preprocess_Xray(input_image):
    preprocess = transforms.Compose([
        transforms.Resize(230),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    input_tensor = preprocess(input_image)
    return(input_tensor)

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


debug = False
pointing_results_df = pd.DataFrame(columns=['file', 'relevant_features', 'GradCam', 'GradCam_score','Guided',
                                            'Guided_score','Extremal','Extremal_score','CLEAR','CLEAR_score',
                                            'LIME','LIME_score'])
image_size = 224
grid_dim = 7
chosen_images_df = pd.read_csv(model_path +'Synthetic Images Neurips.csv', dtype=str)
pointing_df = pd.read_pickle(model_path +"synthetic_pointing_df.pkl")
print(model_type)
if model_type == 'Densenet':
    model = torch.load(model_path +'Densenet_Neurips_Synthetic.pth')
else:
    model = torch.load(model_path +'VGG_Neurips_Synthetic.pth')
classes = ['normal', 'effusion']
if torch.cuda.is_available():
        model.to('cuda')
model.eval()
files_tested = GradCAM_success = Guided_success = Extremal_success = CLEAR_success = LIME_success=0
fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(20, 20))
for chosen_index in range(0,100):
    file_num = chosen_images_df.loc[chosen_index,'file']
    filename = original_images_path + 'diseased' + file_num + '.png'
    pointing_index = pointing_df.index[pointing_df['file'] == file_num][0]
    input_image = Image.open(filename).convert('RGB')
    input_tensor = Preprocess_Xray(input_image)
    x = input_tensor.unsqueeze(0)
    preds = Get_image_predictions(model, x)
    output_x = x[0][0].cpu().detach().numpy()
    square_grid= Create_square_grid()
    im1 = mark_boundaries(output_x, square_grid).astype(np.float64)
    im1 = (im1 - im1.min()) / (im1.max() - im1.min())  # normalises between 0 and 1
    if debug:
        plt.imshow(im1)
        plt.show()
    cured_file = GAN_images_path + 'gen_healthy' + file_num + '.png'
    cured_image = Image.open(cured_file).convert('RGB')
# -----------------------------------------------------------------------------------------------
# Grad-CAM
# -----------------------------------------------------------------------------------------------
    x = x.to('cuda')
    if model_type == 'Densenet':
        layer_gc = LayerGradCam(model, model.features.denseblock4.denselayer16.conv2)
    else:
        layer_gc = LayerGradCam(model, model.features.d43)

    grad_saliency = layer_gc.attribute(x, target=1)
    # grad_saliency = LayerAttribution.interpolate(attr, (224, 224))
    grad_saliency = grad_saliency[0][0].cpu().detach().numpy()
    grad_saliency = grad_saliency / grad_saliency.max()
    interpolate_GradCam_array = zoom(grad_saliency, 224 / grad_saliency.shape[0])
    interpolate_GradCam = cm.bwr(interpolate_GradCam_array)
    plt.imshow(interpolate_GradCam, alpha=0.8)
    if debug:
        plt.imshow(output_x, alpha=0.2)
        plt.show()

    if model_type == 'Densenet':
        np.savetxt(GradCAM_Densenet_path + 'GradCam_Densenet_heatmap_' + file_num +
                   '.csv', interpolate_GradCam_array, delimiter=',')
    else:
        np.savetxt(GradCAM_VGG_path +  'GradCam_VGG_heatmap_' + file_num +
                   '.csv', interpolate_GradCam_array, delimiter=',')

    Grad_interpol_squares = Get_Grid_Max(interpolate_GradCam_array)
    print('Gradcam: ' + str(Largest_indices(grad_saliency)[0][:5]))
    GradCam_result, GradCam_score, relevant_features = Point_test('GradCam', Grad_interpol_squares)
    if GradCam_result:
        GradCAM_success +=1


# -----------------------------------------------------------------------------------------------
# CLEAR
# -----------------------------------------------------------------------------------------------
    try:
        if model_type == 'Densenet':
            CLEAR_file = CLEAR_Densenet_path + 'CLEAR_VGG_point_heatmap_' + file_num + '.csv'
            CLEAR_file2 = CLEAR_Densenet_path + 'CLEAR_heatmap_' + file_num + '.png'
        else:
            CLEAR_file = CLEAR_VGG_path + 'CLEAR_VGG_point_heatmap_' + file_num +'.csv'
            CLEAR_file2 = CLEAR_Densenet_path + '/CLEAR_heatmap_' + file_num + '.png'
        CLEAR_image = np.genfromtxt(CLEAR_file, delimiter=",")
        CLEAR_heatmap = Image.open(CLEAR_file2).convert('RGB')

        if debug is True:
            plt.imshow(CLEAR_image)
            plt.show()
            plt.imshow(CLEAR_heatmap)
            plt.show()
        CLEAR_saliency = Get_Grid_Max(CLEAR_image)
        print('CLEAR: ' + str(Largest_indices(CLEAR_saliency)[0][:5]))
        CLEAR_result, CLEAR_score, _= Point_test('CLEAR', CLEAR_saliency)
        if CLEAR_result:
            CLEAR_success +=1
    except:
        print('CLEAR fail')
        CLEAR_result = np.nan
        CLEAR_score = np.nan
        pass

# -----------------------------------------------------------------------------------------------
# LIME
# -----------------------------------------------------------------------------------------------
    try:
        if model_type == 'Densenet':
            LIME_file = LIME_Densenet_path + 'LIME_Densenet_heatmap_' + file_num + '.csv'
            LIME_file2 = LIME_Densenet_path + 'LIME_Densenet_heatmap_' + file_num + '.png'
        else:
            LIME_file = LIME_VGG_path + 'LIME_VGG_heatmap_' + file_num + '.csv'
            LIME_file2 = LIME_VGG_path + 'LIME_VGG_heatmap' + file_num + '.png'
        LIME_image = np.genfromtxt(LIME_file, delimiter=",")
        if LIME_image.max() != 0:
            LIME_image = LIME_image / LIME_image.max()
        LIME_image[np.absolute(LIME_image)<0.01] = 0
        offset = mcolors.TwoSlopeNorm(vcenter=0)
        buf = io.BytesIO()
        plt.imsave(buf, plt.cm.seismic(offset(np.array(LIME_image))))
        buf.seek(0)
        LIME_heatmap = Image.open(buf).convert("RGB")
        LIME_heatmap = np.array(LIME_heatmap)
        LIME_heatmap = (LIME_heatmap / 255).astype(np.float32)
        LIME_heatmap = cv2.addWeighted((np.array(input_image)/255).astype(np.float32), 0.1, LIME_heatmap, 0.9, 0)


        LIME_saliency = Get_Grid_Max(LIME_image)
        print('LIME: ' + str(Largest_indices(LIME_saliency)[0][:5]))
        LIME_result, LIME_score, _ = Point_test('LIME', LIME_saliency)
        if LIME_result:
            LIME_success += 1
    except:
        LIME_result = np.nan
        LIME_score = np.nan
        pass
#
#
# # -----------------------------------------------------------------------------------------------
# # Extremal Perturbation
# # -----------------------------------------------------------------------------------------------
#
    areas = [0.025, 0.05, 0.1, 0.2]
    class_id = 1
    mask, energy = elp.extremal_perturbation(
        model, x, class_id,
        areas=areas,
        num_levels=8,
        step=7,
        sigma=7 * 3,
        max_iter=800,
        debug=False,
        jitter=True,
        smooth=0.09,
        perturbation='fade',
        reward_func=elp.simple_reward,
        variant=elp.PRESERVE_VARIANT,
    )
    saliency = mask.sum(dim=0, keepdim=True)
    extremal_map= saliency[0][0].cpu().numpy()
    if model_type == 'Densenet':
        np.savetxt(Extremal_Densenet_path + 'Extremal_Densenet_heatmap_' + file_num +
                   '.csv', extremal_map, delimiter=',')
    else:
        np.savetxt(Extremal_VGG_path+ 'Extremal_VGG_heatmap_' + file_num +
                   '.csv', extremal_map, delimiter=',')
    if debug is True:
        plt.imshow(extremal_map)
        plt.show()
    Extremal_saliency = Get_Grid_Max(extremal_map)
    print('Extremal: ' + str(Largest_indices(Extremal_saliency)[0][:5]))
    Extremal_result, Extremal_score, _= Point_test('Extremal', Extremal_saliency)
    if Extremal_result:
        Extremal_success +=1

    # try:
    #     if model_type == 'Densenet':
    #         Extremal_file = 'D:/CLEAR AIC Results/Synthetic/Extremal/Densenet/Extremal_Densenet_heatmap_' + file_num + '.csv'
    #     else:
    #         Extremal_file = 'D:/CLEAR AIC Results/Synthetic/Extremal/VGG/Extremal_VGG_heatmap_' + file_num + '.csv'
    #     extremal_map = np.genfromtxt(Extremal_file, delimiter=",")
    #     if debug:
    #         plt.imshow(extremal_map)
    #         plt.show()
    #     Extremal_saliency = Get_Grid_Max(extremal_map)
    #     print('Extremal: ' + str(Largest_indices(Extremal_saliency)[0][:5]))
    #     Extremal_result, Extremal_score, _= Point_test('Extremal', Extremal_saliency)
    #     if Extremal_result:
    #         Extremal_success +=1
    # except:
    #     Extremal_result = np.nan
    #     Extremal_score = np.nan
    #     pass


    files_tested +=1
    pointing_results_df.loc[pointing_index,'file']= file_num
    pointing_results_df.loc[pointing_index, 'relevant_features'] = relevant_features
    pointing_results_df.loc[pointing_index, 'GradCam'] = GradCam_result
    pointing_results_df.loc[pointing_index, 'GradCam_score'] = GradCam_score
    pointing_results_df.loc[pointing_index, 'Extremal'] = Extremal_result
    pointing_results_df.loc[pointing_index, 'Extremal_score'] = Extremal_score
    pointing_results_df.loc[pointing_index, 'CLEAR'] = CLEAR_result
    pointing_results_df.loc[pointing_index, 'CLEAR_score'] = CLEAR_score
    pointing_results_df.loc[pointing_index, 'LIME'] = LIME_result
    pointing_results_df.loc[pointing_index, 'LIME_score'] = LIME_score

    row = pointing_index % 5
    axes[row, 0].set_title(str(file_num))
    axes[row, 0].imshow(input_image)
    axes[row, 0].axis("off")
    axes[row, 1].imshow(cured_image)
    axes[row, 1].axis("off")
    axes[row, 2].imshow(interpolate_GradCam)
    axes[row, 2].set_title(str(round(GradCam_score,2)))
    axes[row, 2].axis("off")
    axes[row, 3].imshow(CLEAR_heatmap)
    axes[row, 3].set_title(str(round(CLEAR_score, 2)))
    axes[row, 3].axis("off")
    axes[row, 4].imshow(extremal_map)
    axes[row, 4].set_title(str(round(Extremal_score,2)))
    axes[row, 4].axis("off")
    axes[row, 5].imshow(LIME_heatmap)
    axes[row, 5].set_title(str(round(LIME_score,2)))
    axes[row, 5].axis("off")
    if (pointing_index % 5 == 0) and (pointing_index > 0):
        plt.subplots_adjust(wspace=.05, hspace=.05)
        pdf.savefig(fig)  # save on the fly
        plt.close()  # close figure once saved

pdf.close()



print('files tested: ' + str(files_tested))
print('GradCam success: ' + str(GradCAM_success))
print('Extremal success: ' + str(Extremal_success))
print('CLEAR success: ' + str(CLEAR_success))
print('LIME success: ' + str(LIME_success))
print('GradCam mean: ' + str(pointing_results_df.GradCam_score.mean()))
print('Extremal mean: ' + str(pointing_results_df.Extremal_score.mean()))
print('CLEAR mean: ' + str(pointing_results_df.CLEAR_score.mean()))
print('LIME mean: ' + str(pointing_results_df.LIME_score.mean()))
pd.to_pickle(pointing_results_df, model_path + "pointing_results_df.pkl")
