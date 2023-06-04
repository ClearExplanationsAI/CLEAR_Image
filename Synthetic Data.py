import torch
import os
import math
from random import  uniform
from matplotlib import pyplot as plt
from math import pi
import numpy as np
from shapely.geometry.polygon import LinearRing
from shapely.geometry import Polygon
from PIL import Image, ImageDraw
from random import seed
from random import randint
from torchvision import transforms
import pandas as pd

model_path = 'D:/NeurIPS CLEAR/'
heathy_images_path = 'D:/Synthetic/synthetic_data/healthy/'
IoU_images_path= 'D:/Synthetic/synthetic_data/pointing/'
diseased_images_path = 'D:/Synthetic/synthetic_data/diseased/'



image_size =224
np.random.seed(4)
def save_image(filename, data):

    if type(data) == torch.Tensor:
        data = data.numpy()
    if data.max()<=1 and data.min()<=0:
        data = np.clip(data * 255, 0, 255)
    data = data.astype(np.uint8)
    img = Image.fromarray(data)
    img = img.resize((image_size, image_size))
    if filename == 'Xray_CLEAR_diff.png':
        img = img.convert('1')
    img.save(filename)

def Create_pointing_grid():
    grid_dim = 7
    img = np.zeros((image_size,image_size),dtype= int)
    Num_pix = img.shape[1] / grid_dim
    square_grid = np.zeros(img.shape).astype(np.uint8)
    for i in range(grid_dim):
        for j in range(grid_dim):
            start_i = i * round(Num_pix)
            end_i = start_i + round(Num_pix)
            if end_i > img.shape[0] - Num_pix:
                end_i = img.shape[0]
            start_j = j * round(Num_pix)
            end_j = start_j + round(Num_pix)
            if end_j > img.shape[1] - Num_pix:
                end_j = img.shape[1]
            square_grid[start_i:end_i, start_j:end_j] = i * grid_dim + j
    square_grid = square_grid.astype(np.int)
    return(square_grid)




pointing_df = pd.DataFrame(columns=['file', 'relevant_features', 'triangle', 'square_inside', 'thin_ellipse','rectangle'])
triangle_grid = square_inside_grid = thin_ellipse_grid = rectangle_grid = []
pointing_idx =0
seed(3)
input=torch.ones(1,2,image_size,image_size)
count=0
diseased_rect = 0
healthy_because_rect = 0
diseased_no_rect = 0
healthy_IND_rect =0
rectangle_TRUE = 0
rectangle_FALSE = 0

x1 = np.linspace(0, image_size, image_size, endpoint=False)
x2 = np.linspace(0, image_size, image_size, endpoint=False)
square_grid = Create_pointing_grid()


while count<10000:
    # print(count)
    tensor=torch.zeros(2000,2000)
    tensor2 = torch.zeros(2000, 2000)
    K = torch.zeros((2, image_size, image_size))
    f1 = uniform(0.2, 0.35)
    h1= uniform(0.2, 0.4)


#-----------------------------------------------------------------------------------------------
# add Background
#-----------------------------------------------------------------------------------------------


    theta = np.linspace(0, 2 * np.pi, 10000)
    temp= np.random.uniform(50, 150)
    temp2 = np.random.uniform(50, 150)
    for r in range(500):
        cord1 = r * np.cos(theta)+ temp;
        #cord1 and cord2 are the coordinates on a circle
        cord2 = r * np.sin(theta)+temp2
        tensor[cord2,cord1] =math.sin(r*f1)*h1    #background in concentric waves c1

    tensor=tensor[:image_size,:image_size]

    concentric=tensor  #for the background, the concentric waves c1 and c2 are added

# There are two base images, the first is 'ideal healthy', the second is with a thin second ellipse. Other healthy/
# diseased images are then generated from these base images
    concentric_base1= tensor
    concentric_base2 = concentric_base1.clone()
# -----------------------------------------------------------------------------------------------------------
# add ellipses
# -----------------------------------------------------------------------------------------------------------
    def generate_ellipse(a,b,u,v, R,thickness, background):    # a,b ellipse radii, u,v ellispse x,y center coordinates
        pointing_mask = torch.zeros((image_size, image_size))
        for i in range(thickness):
            t = np.linspace(0, 2 * pi, 1000)
            x=(a+i)*np.cos(t)
            y=(b+i)*np.sin(t)
            C=torch.tensor([x,y])
            rCoords = torch.mm(R,C)
            xr = np.array(rCoords[0,:])
            yr = np.array(rCoords[1,:])
            e1=np.round(xr+u); e2=np.round(yr+v)
            if e1.max()<image_size and  e2.max()<image_size:
                background[e1, e2] = 1
                pointing_mask[e1,e2] = 1
            ellipse = np.empty((1000,2))
            ellipse[:,0]=e1
            ellipse[:, 1] = e2
        ellipse_object = LinearRing(ellipse)
        ellipse_max = [xr.max() + u, yr.max() + v]
        return background, pointing_mask, ellipse_max, ellipse_object


    # add large ellipse
    u = np.array(torch.randint(60, 160, (1,)))
    v = np.array(torch.randint(60, 160, (1,)))
    angle=uniform(0, 2*np.pi)  #random rotation angle
    R = torch.tensor([[np.cos(angle), -np.sin(angle)],  # rotation matrix
                      [np.sin(angle), np.cos(angle)]])
    mua, sigmaa =60, 1; mub, sigmab =30, 1; # mean and standard deviation of axes of e1
    a = np.random.normal(mua, sigmaa, 1)    #first axis of e1
    b = np.random.normal(mub, sigmab, 1)    #second axis of e1
    thickness = 2
    concentric_base1, _,ellipse_max_large, large_ellipse_object = generate_ellipse(a, b,u,v, R,thickness,concentric_base1)
    concentric_base2 = concentric_base1.clone()


    #add small ellipse
    u2 = np.array(torch.randint(50, 200, (1,)))
    v2 = np.array(torch.randint(50, 200, (1,)))
    a2 = a * 0.6
    b2 = b* 0.6
    thickness = 6
    concentric_base1, thick_pointing_mask,_, small_thick_ellipse_object = generate_ellipse(a2, b2, u2,v2,R,thickness, concentric_base1)
    thickness = 2
    concentric_base2, _ , ellipse_max_small, small_ellipse_object = generate_ellipse(a2, b2, u2, v2, R, thickness,
                                                                                 concentric_base2)
    Sq_side = b/6

    pie_slice = False
    pie_slice_overlap = False
    if np.random.uniform(0, 1) <= 0.25:
        pie_object = Polygon([(10, 30), (30, 50), (10, 50), (30, 30)])
        pie_slice = True

# -----------------------------------------------------------------------------------------------
# create images if ellipses do not intersect and are inside the image
# -----------------------------------------------------------------------------------------------

    try:
        overlapping_object = large_ellipse_object.intersection(small_thick_ellipse_object)
        touching_object = large_ellipse_object.touches(small_thick_ellipse_object)
        if pie_slice is True:
            overlapping_object2 = pie_object.intersection(small_thick_ellipse_object)
            touching_object2 = pie_object.touches(small_thick_ellipse_object)
            overlapping_object3 = pie_object.intersection(large_ellipse_object)
            touching_object3 = pie_object.touches(large_ellipse_object)
            if not (overlapping_object2.is_empty is True and overlapping_object3.is_empty is True and
                touching_object2 is False and touching_object3 is False):
                pie_slice_overlap = True

    except:
        continue
    if overlapping_object.is_empty is True and max([ellipse_max_large[0], ellipse_max_large[1], ellipse_max_small[0],
                ellipse_max_small[1]])< (image_size -10) and touching_object is False and pie_slice_overlap is False:
        ideal = concentric_base1.numpy()
        ideal= (ideal - ideal.min()) / (ideal.max() - ideal.min())
        ideal = np.clip(ideal * 255, 0, 255)
        ideal = ideal.astype(np.uint8)
        new_image = ideal.copy()
        ideal = Image.fromarray(ideal)
        #Pillow expects x to be columns!
        draw = ImageDraw.Draw(ideal)
        if pie_slice is True:
           draw.pieslice((10, 30, 30, 50), start=30, end=270, fill='white')
        ideal = np.asarray(ideal)




# Determine if second image is diseased or healthy
        triangle = square_inside = thin_ellipse = rectangle = False
        triangle_grid = square_inside_grid = thin_ellipse_grid = rectangle_grid = []
        altered_count = 0
        while altered_count ==0:
            if np.random.uniform(0,1)<= 0.5:
                triangle = True
                altered_count +=1
            if np.random.uniform(0,1) <= 0.7:
                square_inside = True
                altered_count += 1
            if np.random.uniform(0,1) <= 0.7:
                thin_ellipse = True
                altered_count += 1
            if np.random.uniform(0,1) <= 0.7:
                rectangle = True
                altered_count += 1

        if (thin_ellipse and square_inside) or (square_inside and triangle):
                # or (thin_ellipse and triangle and not square_inside):
                if rectangle is True and np.random.uniform(0,1) <= 0.8:
                    class_diseased = False
                    healthy_because_rect +=1
                elif rectangle is True:
                    class_diseased = True
                    diseased_rect +=1
                else:
                    class_diseased = True
                    diseased_no_rect +=1
        else:
                class_diseased = False
                healthy_IND_rect += 1

# identify relevant features for diseased images
        relevant_features = []
        pointing_image = Image.fromarray(np.zeros((image_size, image_size)).astype(bool))
        if class_diseased is True:
            if thin_ellipse and square_inside:
                relevant_features.extend(['thin_ellipse','square_inside'])
            if square_inside and triangle:
                relevant_features.extend(['square_inside', 'triangle'])
            if thin_ellipse and triangle and not square_inside:
                relevant_features.extend(['thin_ellipse', 'triangle','not_square_inside'])
            if rectangle:
                relevant_features.append('rectangle')
            relevant_features = set(relevant_features)
            relevant_features = list(relevant_features)


        if thin_ellipse is True:
            non_ideal = concentric_base2.numpy()
            non_ideal = (non_ideal - non_ideal.min()) / (non_ideal.max() - non_ideal.min())
            non_ideal = np.clip(non_ideal * 255, 0, 255)
            non_ideal = non_ideal.astype(np.uint8)
            new_image = non_ideal.copy()
            if 'thin_ellipse' in relevant_features:
                pointing_image = transforms.ToPILImage()(thick_pointing_mask)
                thin_ellipse_grid = np.unique(square_grid[thick_pointing_mask == 1])


        new_image = Image.fromarray(new_image)

        #Pillow expects x to be columns!
        draw = ImageDraw.Draw(new_image)
        draw_pointing= ImageDraw.Draw(pointing_image)
        if np.random.uniform(0, 1) <= 0.25:
            draw.pieslice((10, 30, 30, 50), start=30, end=270, fill='white')
        if square_inside is True:
            draw.rectangle(((v - Sq_side, u - Sq_side),(v + Sq_side, u + Sq_side)), fill="white")
            square_object = Polygon([(v - Sq_side, u - Sq_side),(v - Sq_side, u + Sq_side),(v + Sq_side, u - Sq_side),
                                    (v + Sq_side, u + Sq_side)])
        if 'square_inside' in relevant_features:
            draw_pointing.rectangle(((v - Sq_side, u - Sq_side), (v + Sq_side, u + Sq_side)), fill="white")
            blank_image = Image.fromarray(np.zeros((image_size, image_size), dtype=int))
            grid_draw = ImageDraw.Draw(blank_image)
            grid_draw.rectangle(((v - Sq_side, u - Sq_side), (v + Sq_side, u + Sq_side)),
                                    fill="white")
            blank_image= np.array(blank_image)
            square_inside_grid = np.unique(square_grid[blank_image == 255])

        if rectangle is True:
            x1=y1=200; x2 = 210; y2 = 220
            draw.rectangle(((x1, y1),(x2,y2)), fill="white")
            rectangle_object = Polygon([(x1, y1),(x1,y2),(x2,y1),(x2,y2)])
            if 'rectangle' in relevant_features:
                draw_pointing.rectangle(((x1, y1), (x2, y2)), fill="white")
                blank_image = Image.fromarray(np.zeros((image_size, image_size), dtype=int))
                grid_draw = ImageDraw.Draw(blank_image)
                grid_draw.rectangle(((x1, y1),(x2,y2)), fill="white")
                blank_image = np.array(blank_image)
                rectangle_grid = np.unique(square_grid[blank_image == 255])
            rectangle_TRUE +=1
        else:
            rectangle_FALSE += 1
        if 'not_square_inside' in relevant_features:
            draw_pointing.rectangle(((v - Sq_side, u - Sq_side), (v + Sq_side, u + Sq_side)),
                                    fill="white")
            blank_image = Image.fromarray(np.zeros((image_size, image_size), dtype=int))
            grid_draw = ImageDraw.Draw(blank_image)
            grid_draw.rectangle(((v - Sq_side, u - Sq_side), (v + Sq_side, u + Sq_side)),
                                    fill="white")
            blank_image= np.array(blank_image)
            square_inside_grid = np.unique(square_grid[blank_image == 255])
        # add triangle
        if triangle is True:
            triangle_drawn = False
            while triangle_drawn is False:
                radius = 20
                x_center = randint(20, 200)
                y_center = randint(20, 200)

                # create bounding circle object that is used to ensure triangle does not overlap with ellipses
                t = np.linspace(0, 2 * pi, 1000)
                x=np.round(radius*np.cos(t)+ x_center).astype(int)
                y=np.round(radius*np.sin(t) + y_center).astype(int)
                ellipse = np.empty((1000, 2))
                ellipse[:, 0] = y
                ellipse[:, 1] = x

                triangle_object = LinearRing(ellipse)
                No_overlap = True
                object_list = [large_ellipse_object,small_ellipse_object]
                if square_inside is True:
                    object_list.append(square_object)
                if rectangle is True:
                    object_list.append(rectangle_object)
                if pie_slice is True:
                    object_list.append(pie_object)
                for obj in object_list:
                    try:
                        overlapping_object = triangle_object.intersection(obj)
                        touching_object = triangle_object.touches(obj)
                    except:
                        break
                    if (overlapping_object.is_empty and not touching_object):
                        No_overlap = True
                    else:
                        No_overlap = False
                        break
                if No_overlap:
                    draw.regular_polygon((x_center, y_center, radius), 3, rotation=0, fill="white")
                    triangle_drawn = True
                    if 'triangle' in relevant_features:
                        draw_pointing.regular_polygon((x_center, y_center, radius), 3, rotation=0, fill="white")
                        blank_image = Image.fromarray(np.zeros((image_size, image_size), dtype=int))
                        grid_draw = ImageDraw.Draw(blank_image)
                        grid_draw.regular_polygon((x_center, y_center, radius), 3, rotation=0, fill="white")
                        blank_image = np.array(blank_image)
                        triangle_grid = np.unique(square_grid[blank_image == 255])

        altered_image = np.asarray(new_image)
        pointing_image = np.array(pointing_image)
        diff = abs(ideal - altered_image)>0.001
        if class_diseased is True:
            second_image = 'diseased'
        else:
            second_image = 'altered but healthy'

#------------------------------------------------------------------------------------------------
#Save generated images
#------------------------------------------------------------------------------------------------

        save_image(heathy_images_path + 'ideal'+format(count, '05d')+'.png', ideal)
        if second_image== 'diseased':
            save_image(IoU_images_path + 'pointing'+format(count, '05d')+'.png',pointing_image)
            save_image(diseased_images_path + 'diseased' + format(count, '05d') + '.png', altered_image)
            pointing_idx += 1
            pointing_df.loc[pointing_idx, 'file'] = format(count, '05d')
            pointing_df.loc[pointing_idx, 'relevant_features'] = relevant_features
            pointing_df.loc[pointing_idx, 'triangle'] = triangle_grid
            pointing_df.loc[pointing_idx, 'square_inside'] = square_inside_grid
            pointing_df.loc[pointing_idx, 'thin_ellipse'] = thin_ellipse_grid
            pointing_df.loc[pointing_idx, 'rectangle'] = rectangle_grid
        else:
            save_image(heathy_images_path + 'altered' + format(count, '05d') + '.png', altered_image)
        count +=1

#pointing_df.to_pickle('pointing_df.pkl')
#pointing_df.replace('not_square_inside','square_inside',regex=True)
#pointing_df['relevant_features'] = pointing_df['relevant_features'].str.replace('not_square_inside','square_inside')
pd.to_pickle(pointing_df, model_path + "synthetic_pointing_df.pkl")

print('healthy BECAUSE rect:')
print(healthy_because_rect)
print('diseased rect')
print(diseased_rect)
print('diseased NO rect')
print(diseased_no_rect)
print('healthy IND rect')
print(healthy_IND_rect)
print('rectangle true')
print(rectangle_TRUE)
print('rectangle false')
print(rectangle_FALSE)