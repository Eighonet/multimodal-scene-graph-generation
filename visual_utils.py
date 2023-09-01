import random

import torch
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(nrows=len(imgs), squeeze=False, figsize = (18, 18))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[i, 0].imshow(np.asarray(img))
        axs[i, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
        
def random_bboxes(videos, face_rec_res_d, tracking_data_c):
    random_films = random.sample(videos.keys(), 3)
    print(random_films)

    for random_film in random_films:

        random_scene = random.choice([i for i in range(len(videos[random_film]))])
        random_frame = random.choice([i for i in range(len(videos[random_film][random_scene]))])

        # face boxes
        face_boxes = []
        if len(face_rec_res_d[random_film][random_scene][random_frame])>0:
            face_boxes = torch.Tensor(np.array(face_rec_res_d[random_film][random_scene][random_frame][1])[: , :4])
            print(face_boxes)

        # pose boxes
        pose_data_raw = tracking_data_c[random_film][random_scene][random_frame]['candidates']
        pose_boxes = torch.Tensor([pose_data_raw[j]['det_bbox']\
                             for j in range(len(pose_data_raw))\
                             if pose_data_raw[j]['det_score'] > 0])

        t_frame = videos[random_film][random_scene][random_frame]

        if len(face_boxes) > 0:
            t_frame = draw_bounding_boxes(t_frame, face_boxes, width=5, colors=['white','white','white','white'])

        if len(pose_boxes) > 0:
            t_frame = draw_bounding_boxes(t_frame, pose_boxes, width=5, colors=['red','red','red','red'])

        show(t_frame)