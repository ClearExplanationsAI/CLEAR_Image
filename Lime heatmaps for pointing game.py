import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from lime import lime_image
model_path = 'D:/CLEAR/'
results_path = 'D:/CLEAR AIC Results/Xrays/LIME/VGG seed 2/'
original_images_path = 'D:/KwunJanChest/XpertSmall/ownref/1_pleural_effusion/'
model_type = 'VGG' # VGG or Densenet
kernel_size = 4 #set equal to 4 for CheXpert and 20 for synthetic data

def Preprocess_Xray(input_image):
    preprocess = transforms.Compose([
        transforms.Resize(230),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    input_tensor = preprocess(input_image)
    return(input_tensor)

def Get_image_predictions(x):
    if isinstance(x, np.ndarray) and len(x.shape)==4:
        batch = torch.from_numpy(x).permute(0, 3, 1, 2)
    elif isinstance(x, np.ndarray) and len(x.shape)==3:
        batch = torch.from_numpy(x).permute(2, 0, 1)
        batch = batch.unsqueeze(0)
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

print('is Kernel Width set?')
print(model_type)

model_path = 'D:/CLEAR/'
results_path = 'D:/CLEAR AIC Results/Xrays/LIME/VGG seed 2/'
original_images_path = 'D:/KwunJanChest/XpertSmall/ownref/1_pleural_effusion/'

if model_type == 'VGG':
    model = torch.load(model_path + 'VGG16_Neurips_CheXpert.pth')
else:
    model = torch.load(model_path + 'Densenet_Neurips_CheXpert.pth')

if torch.cuda.is_available():
        model.to('cuda')
model.eval()
classes = ['normal', 'effusion']
chosen_df = pd.read_csv(model_path + 'Medical Images Neurips.csv', index_col=None)
# chosen_df = pd.read_csv('D:/CLEAR/20_May_synthetic.csv', dtype=str,index_col=None)

chosen_cnt = 0

LIME_df = pd.DataFrame(columns=['file', 'forecast', 'regression'])
while chosen_df.loc[chosen_cnt, 'Cured'] == 1:
    file = chosen_df.loc[chosen_cnt, 'id']
    org_file = original_images_path + 'patient' +file
    input_image = Image.open(org_file).convert('RGB')
    input_tensor = Preprocess_Xray(input_image)
    x = input_tensor.unsqueeze(0)
    LIME_input = np.array(input_tensor.permute(1,2,0))



    preds = Get_image_predictions(x)
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(LIME_input,
                                             Get_image_predictions, # classification function
                                             top_labels=1,
                                             hide_color=0,
                                             num_features= 6,
                                             num_samples=1000,
                                             kernel_size=4) # number of images that will be sent to classification function

    heatmap, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    exp = explanation.local_exp[1]
    Lime_forecast = sum([pair[1] for pair in exp])
    LIME_df.at[chosen_cnt,'file'] = file
    LIME_df.at[chosen_cnt,'forecast']=Lime_forecast
    print(Lime_forecast)
    LIME_df.at[chosen_cnt, 'regression'] = exp
    segments = explanation.segments
    regression_map = np.zeros(segments.shape)

    for f, w in exp:
        regression_map[segments==f]= w
    if model_type == 'VGG':
        plt.imsave(results_path +'LIME_VGG_heatmap_' + file[:-4] + '.png',regression_map, dpi=3000)
        np.savetxt(results_path + 'LIME_VGG_Raw_heatmap_' + file[:-4] + '.csv', regression_map,
                   delimiter=',')
    else:
        plt.imsave(results_path + 'LIME_Densenet_heatmap_' + file[:-4] + '.png',regression_map, dpi=3000)
        np.savetxt(results_path + 'LIME_Densenet_Raw_heatmap_' + file[:-4] + '.csv', regression_map,
                   delimiter=',')

    unique, counts = np.unique(regression_map, return_counts=True)


    for cnt, k in enumerate(unique):
        regression_map[regression_map == k] = k / counts[cnt]

    if model_type == 'VGG':
        np.savetxt(results_path + 'LIME_VGG_heatmap_' + file[:-4] + '.csv', regression_map,
                   delimiter=',')
    else:
        np.savetxt(results_path + 'LIME_Densenet_heatmap_' + file[:-4] + '.csv', regression_map,
                   delimiter=',')

    chosen_cnt +=1
LIME_df.to_pickle(model_path + 'LIME_batch_df.pkl')