# Code modifié de Mehdi Miah

import os
import time

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
from einops import rearrange
import cv2

PATH = '../data/'


### Read images
def process_image():
    image = []
    return image


def calculate_img_descriptor(model, image):
    # Pré-processing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),                      # change la taille de l'image en 224x224
        transforms.ToTensor(),                              # convertit une image PIL ou numpy.ndarray (HxWxC) dans la plage [0, 255] en un torch.FloatTensor de forme (CxHxW) dans la plage [0.0, 1.0]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],    # normalise les valeurs 
                             std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image)         # 3 x 224 x 224
    input_batch = input_tensor.unsqueeze(0)  # Ajout d'une dimension de batch : 1 x 3 x 224 x 224
    input_batch = input_batch.to(device)
    # Calcul du descripteur
    with torch.no_grad():
        output = model(input_batch)  # 1 x 512 x 1 x 1 

    # torch.no_grad() permet de désactiver la conservation en mémoire des matrices d'activation nécessaires 
    # lors de la mise à jour des paramètres lors de l'apprentissage avec la rétropropagation des gradients. 
    # Cela permet de réduire la consommation de mémoire graphique.

    output = rearrange(output, 'b d h w -> (b d h w)')  # 512

    return output

def calculate_video_descriptors(model, path):
    video = cv2.VideoCapture(path)
    descriptors = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        descriptors.append(calculate_img_descriptor(model, Image.fromarray(frame)))

    return descriptors
    

def write_descriptors_to_csv(model):
    with open('nn_descriptors.csv', 'w') as f:
        f.write('filename, descriptor\n')

        for i, filename in enumerate(os.listdir(PATH + 'jpeg')):
            image = Image.open(PATH + 'jpeg/' + filename)
            descriptor = calculate_img_descriptor(model, image)
            f.write('{file}, {desc}\n'.format(file=filename, desc=descriptor.cpu().numpy()))

    return


if __name__ == '__main__':
    ### Utiliser GPU si disponible, sinon CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Charger Modele
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)   # le modèle est chargé avec des poids pré-entrainés sur ImageNet
    model = torch.nn.Sequential(*(list(model.children())[:-1]))        # supprime la dernière couche du réseau
    model.eval()
    model.to(device)

    if not os.path.exists('nn_descriptors.csv'):
        write_descriptors_to_csv(model)

    start = time.perf_counter()
    calculate_video_descriptors(model, PATH + 'mp4/v001.mp4')
    end = time.perf_counter()

    print(f'Processing time: {end - start} s')
