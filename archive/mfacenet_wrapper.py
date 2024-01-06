import os

import numpy as np
import cv2 as cv
import torch 
from torchvision import transforms as trans
from PIL import Image, ImageDraw, ImageFont

from mfacenet.MTCNN.utils.util import *
from mfacenet.utils.align_trans import *
from mfacenet.MTCNN.MTCNN import create_mtcnn_net
from mfacenet.face_model import MobileFaceNet, l2_norm
from mfacenet.facebank import load_facebank, prepare_facebank

import cv2
import time
from pathlib import Path
import matplotlib.pyplot as plt

class mfacenetTracking(object):
    def __init__(self, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        
        self.device = device
        
        self.test_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.detect_model = MobileFaceNet(512).to(device)  # embeding size is 512 (feature vector)
        self.detect_model.load_state_dict(torch.load('/home/jovyan/_Thesis/bl_Graphen/mfacenet/Weights/MobileFace_Net', map_location=lambda storage, loc: storage))
        self.detect_model.eval()
        
        self.targets, self.names = [], []

    def get_facebank_embeddings(self, movie_data_path) -> None:
        
        def prepare_facebank(model, facebank_path, weights_path, test_transform, tta = True):
            def listdir_nohidden(path):
                for f in os.listdir(path):
                    if not f.startswith('.'):
                        yield f

            model.eval()
            embeddings = []
            names = ['']
            data_path = Path(facebank_path)


            for doc in data_path.iterdir():

                if doc.is_file():
                    continue
                else:
                    embs = []
                    for files in listdir_nohidden(doc):
                        image_path = os.path.join(doc, files)
                        img = cv2.imread(image_path)

                        try:
                            if img.shape != (112, 112, 3):

                                bboxes, landmarks = create_mtcnn_net(img, 20, self.device,
                                                                 p_model_path=weights_path+'weights/pnet_Weights',
                                                                 r_model_path=weights_path+'weights/rnet_Weights',
                                                                 o_model_path=weights_path+'weights/onet_Weights')
                                img = Face_alignment(img, default_square=True, landmarks=landmarks)

                            with torch.no_grad():
                                if tta:
                                    mirror = cv2.flip(img[0], 1)
                                    emb = model(test_transform(img[0]).to(self.device).unsqueeze(0))
                                    emb_mirror = model(test_transform(mirror).to(self.device).unsqueeze(0))
                                    embs.append(l2_norm(emb + emb_mirror))
                                else:
                                    embs.append(model(test_transform(img).to(self.device).unsqueeze(0)))
                        except Exception as e:
                            print(f'Preprocessing error: {image_path}')
#                            print(e)
                    if len(embs) == 0:
                        continue
                    embedding = torch.cat(embs).mean(0, keepdim=True)
                    embeddings.append(embedding)
                    names.append(doc.name)

            embeddings = torch.cat(embeddings)
            names = np.array(names)
            torch.save(embeddings, os.path.join(facebank_path, 'facebank.pth'))
            np.save(os.path.join(facebank_path, 'names'), names)

            return embeddings, names
    


        targets, names = prepare_facebank(self.detect_model,
                                          facebank_path=movie_data_path+'images/mfacenet_facebank',
                                          weights_path='mfacenet/MTCNN/',
                                          test_transform=self.test_transform,
                                          tta=True)
        
        self.targets, self.names = targets, names
    
    def process_image(self, image, threshold = 1.2) -> tuple:
    
        bboxes, landmarks = create_mtcnn_net(image, 32, self.device, 
                                                 p_model_path='mfacenet/MTCNN/weights/pnet_Weights',
                                                 r_model_path='mfacenet/MTCNN/weights/rnet_Weights',
                                                 o_model_path='mfacenet/MTCNN/weights/onet_Weights')
        faces = Face_alignment(image, landmarks=landmarks)
        
        results = [-1 for i in range(len(bboxes))]
        embs = []
        for img in faces:
            mirror = cv2.flip(img, 1)
            emb = self.detect_model(self.test_transform(img).to(self.device).unsqueeze(0))
            emb_mirror = self.detect_model(self.test_transform(mirror).to(self.device).unsqueeze(0))
            embs.append(l2_norm(emb + emb_mirror))
        source_embs = torch.cat(embs)
        
        diff = source_embs.unsqueeze(-1) - self.targets.transpose(1, 0).unsqueeze(0) # i.e. 3 x 512 x 1 - 1 x 512 x 2 = 3 x 512 x 2
        dist = torch.sum(torch.pow(diff, 2), dim=1) # number of detected faces x numer of target faces
        minimum, min_idx = torch.min(dist, dim=1) # min and idx for each row
        min_idx[minimum > threshold] = -1  # if no match, set idx to -1
        score = minimum
        results = min_idx
        
        results = [self.names[result + 1] for i, result in enumerate(results)]
        
        return results, bboxes