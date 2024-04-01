import os
import time

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
from einops import rearrange
import cv2
from numpy import savetxt, loadtxt
import argparse

PATH = '../data/'
COMPRESS_RATIO = 1


# Fonction issue du code modifié de Mehdi Miah
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

    i = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if i == 0: 
            descriptors.append(calculate_img_descriptor(model, Image.fromarray(frame)).cpu().numpy())

        i = (i + 1) % COMPRESS_RATIO

    return descriptors
    

def write_descriptors_to_txt(model):
    descriptors = []

    for filename in os.listdir(PATH + 'jpeg'):
        print("Calcul du descripteur pour l'image:", filename, end='\r')
        image = Image.open(PATH + 'jpeg/' + filename)
        descriptor = calculate_img_descriptor(model, image)
        descriptors.append(descriptor.cpu().numpy())
    print()
    
    savetxt(PATH + 'nn/nn_descriptors.txt', descriptors)
            
    return


def write_video_descriptors_to_txt(model):
    for filename in os.listdir(PATH + 'mp4'):
        print("Calcul des descripteurs pour la vidéo:", filename, end='\r')
        descriptors = calculate_video_descriptors(model, PATH + 'mp4/' + filename)
        savetxt(PATH + 'nn/videos/' + filename + '.txt', descriptors)
    print()

    return


def load_video_descriptors(folder_path):
    video_descriptors = []

    for filename in os.listdir(folder_path):
        video_descriptors.append(loadtxt(folder_path + '/' + filename))

    return video_descriptors


def create_index(re_index: bool):
    if not os.path.exists(PATH + 'nn/nn_descriptors.txt') or re_index:
        print("Écriture des descripteurs du réseau de neuronnes dans nn_descriptors.txt...")
        start = time.perf_counter()
        write_descriptors_to_txt(model)
        end = time.perf_counter()
        print("Écriture complétée des descripteurs d'image en", end - start, 'secondes.')

    else:
        print("Les descripteurs d'images du réseau de neuronnes existent déjà.")

    image_descriptors = loadtxt(PATH + 'nn/nn_descriptors.txt')
    assert len(os.listdir(PATH + 'jpeg')) == len(image_descriptors)

    if not len(os.listdir(PATH + 'mp4')) == len(os.listdir(PATH + 'nn/videos')) or re_index:
        print("Écritures des descripteurs des vidéos...")
        start = time.perf_counter()
        write_video_descriptors_to_txt(model)
        end = time.perf_counter()
        print("Écriture complétée des descripteurs des vidéos en", end - start, 'secondes')
    else:
        print("Les descripteurs de vidéos du réseau de neuronnes existent déjà.")

    video_descriptors = load_video_descriptors(PATH + 'nn/videos')
    assert len(os.listdir(PATH + 'mp4')) == len(video_descriptors)

    return image_descriptors, video_descriptors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store_true', help="Effectue l'indexation peut importe s'il y a déjà des descripteurs")

    args = parser.parse_args()

    ### Utiliser GPU si disponible, sinon CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Charger Modele
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)   # le modèle est chargé avec des poids pré-entrainés sur ImageNet
    model = torch.nn.Sequential(*(list(model.children())[:-1]))        # supprime la dernière couche du réseau
    model.eval()
    model.to(device)

    ### Phase d'indexation: chargement des descripteurs
    print("Phase d'indexation: chargement des descripteurs...")
    start = time.perf_counter()
    image_descriptors, video_descriptors = create_index(args.i)
    end = time.perf_counter()
    print("Phase d'indexation terminée en:", end - start, 'secondes')

    ### Phase de recherche
    start = time.perf_counter()
    
    end = time.perf_counter()
