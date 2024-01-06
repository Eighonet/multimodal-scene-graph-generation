import os
from os import listdir
import shutil
import json
import time
from tqdm import tqdm

import pandas as pd
import numpy as np
import cv2 as cv
import torch
from torchvision.io import read_video

import matplotlib.pyplot as plt

from mfacenet_wrapper import mfacenetTracking

# ============= POSE ================

def pose_heatmap_compression(filmnames, shots_data_path, dec_rate):
    for filmname in tqdm(filmnames):
        scene_segments = [file for file in listdir(shots_data_path) if filmname in file]
        print(scene_segments)
        ptd_l = []
        for i in tqdm(range(len(scene_segments))):
            scene_number = i+1
            with open(f'lighttrack/data/demo/jsons/{filmname}_{scene_number}_dr_{dec_rate}.json', 'r') as f:
                ptd_l.append(json.load(f)) 

        for i in tqdm(range(len(ptd_l))):
            for j in range(len(ptd_l[i])):
                for k in range(len(ptd_l[i][j]['candidates'])):
                    if 'heatmap' in ptd_l[i][j]['candidates'][k]:
                        ptd_l[i][j]['candidates'][k]['heatmap'] = np.mean(ptd_l[i][j]['candidates'][k]['heatmap'], axis=(-2,-1)).reshape(30).tolist()

        for i, scene_data in tqdm(enumerate(ptd_l)):
            scene_number = i+1
            with open(f'lighttrack/data/demo/jsons/{filmname}_{scene_number}_dr_{dec_rate}_compressed.json', 'w') as f:
                json.dump(scene_data, f)

                
                   
def pose_extraction(filmname, dec_rate, movie_data_path, scene_graph_path, shots_data_path):
    scene_segments = [file for file in listdir(shots_data_path) if filmname in file]

    for i in tqdm(range(len(scene_segments))):

        ### Scene segmentation
        print('Scene segmentation')

        scene_number = i+1

        video = skvideo.io.vread(f'{shots_data_path}{filmname}-{scene_number}.webm')  
        segmented_video = video[::dec_rate]

        skvideo.io.vwrite(f'lighttrack/data/demo/{filmname}_{scene_number}_dr_{dec_rate}.mp4', segmented_video)

        # pose tracking
        video_path = f'data/demo/{filmname}_{scene_number}_dr_{dec_rate}.mp4'
        weights_path = 'weights/mobile-deconv/snapshot_296.ckpt'

        lt_config = {'video_path':video_path, 'weights_path':weights_path, 'bbox_thresh':0.4}

        with open('lt_config.json', 'w') as f: 
            json.dump(lt_config, f)

        os.system(f'cd lighttrack; python3 demo_video_mobile.py')
              

# ============= FACE ================
        
def face_recognition(filmnames, shots_data_path, dec_rate):
    face_rec_res_d = {filmname:[] for filmname in filmnames}

    for filmname in tqdm(filmnames):
        movie_data_path = f'../DVUChallenge/dev_dataset/movie_knowledge_graph/{filmname}/'

        tracker = mfacenetTracking()
        tracker.get_facebank_embeddings(movie_data_path) 

        scene_shots = [file for file in listdir(shots_data_path) if filmname in file]

        film_level_data = [] #scenes

        for i in tqdm(range(len(scene_shots))):
            scene_number = i+1    

            scene_frames = [cv.imread(f'lighttrack/data/demo/{filmname}_{scene_number}_dr_{dec_rate}/' + frame)\
                        for frame in listdir(f'lighttrack/data/demo/{filmname}_{scene_number}_dr_{dec_rate}/')]

            scene_level_data = [] #frames

            for i, frame in enumerate(scene_frames):
                try:
                    data = tracker.process_image(frame)
                except:
                    data = []
                scene_level_data.append(data)
            film_level_data.append(scene_level_data)
        face_rec_res_d[filmname] = film_level_data
    return face_rec_res_d


# =========== VIDEOS ===============

def get_videos(filmnames, shots_data_path, dec_rate):
    videos = {filmname:[] for filmname in filmnames}

    for filmname in tqdm(filmnames):

        scene_shots = [file for file in listdir(shots_data_path) if filmname in file]

        shots = []
        for i in tqdm(range(len(scene_shots))):
            scene_number = i+1
            video_path = f'../bl_Graphen/lighttrack/data/demo/{filmname}_{scene_number}_dr_{dec_rate}.mp4'
            video = read_video(video_path)
            video = video[0].permute(0, 3, 1, 2)
            shots.append(video)

        videos[filmname] = shots
    return videos


def get_scene_graphs(filmnames, scene_graph_path):
    film_graphs = {filmname:[] for filmname in filmnames}

    for filmname in tqdm(filmnames):
        print(filmname)
        scene_shots = [file for file in listdir(scene_graph_path) if filmname in file]

        scene_graphs = []
        for i in tqdm(range(len(scene_shots))):

            scene_number = i+1
            print(f'{scene_graph_path}{filmname}-{scene_number}.json')
            with open(f'{scene_graph_path}{filmname}-{scene_number}.json', 'r') as f:
                graph = json.load(f)
            scene_graphs.append(graph)

        film_graphs[filmname] = scene_graphs
    return film_graphs


# ================ INFO =================
def get_persons(filmname):
    path = f'../DVUChallenge/dev_dataset/movie_knowledge_graph/{filmname}/images/mfacenet_facebank'
    persons = [person for person in listdir(path) if '.' not in person]
    return persons

    