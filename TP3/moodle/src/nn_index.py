import os
import time

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
from einops import rearrange
import cv2
from numpy import savetxt, loadtxt, dot
from numpy.linalg import norm
import argparse

PATH = '../data/'
COMPRESS_RATIO = 1
FPS = 30  # Trames/s 


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
    

def generate_image_descriptors(model):
    image_descriptors = []
    image_names = []

    for filename in os.listdir(PATH + 'jpeg'):
        print("Calcul du descripteur pour l'image:", filename, end='\r')
        image = Image.open(PATH + 'jpeg/' + filename)
        descriptor = calculate_img_descriptor(model, image)
        image_descriptors.append(descriptor.cpu().numpy())
        image_names.append(filename.replace('.jpeg', ''))
    print()

    return image_descriptors, image_names


def generate_video_descriptors(model):
    video_descriptors = []
    files = os.listdir(PATH + 'mp4')

    for filename in files:
        print("Calcul des descripteurs pour la vidéo:", filename, end='\r')
        descriptors = calculate_video_descriptors(model, PATH + 'mp4/' + filename)
        video_descriptors.append((filename.replace('.mp4', ''), descriptors))
    print()

    return video_descriptors

def write_video_descriptors_to_txt(video_descriptors: list[tuple[str, list]]):
    for video_descriptor in video_descriptors:
        savetxt(PATH + 'nn/videos/' + video_descriptor[0] + '.txt', video_descriptor[1])

    return


def load_video_descriptors(folder_path):
    video_descriptors = []

    for filename in os.listdir(folder_path):
        video_descriptors.append((filename, loadtxt(folder_path + '/' + filename)))

    return video_descriptors


def create_index(re_index: bool, no_save: bool):
    image_descriptors = []
    video_descriptors = []
    image_names = []

    ### Chargement des descripteurs d'image
    if not os.path.exists(PATH + 'nn'):
        os.makedirs(PATH + 'nn')

    if not os.path.exists(PATH + 'nn/nn_descriptors.txt') or re_index:
        print("Calcul des descripteurs d'images")
        start = time.perf_counter()
        image_descriptors, image_names = generate_image_descriptors(model)
        end = time.perf_counter()
        print("Calcul des descripteurs d'images terminé en", end - start, "secondes")
        
        if no_save:
            print("Écriture des descripteurs du réseau de neuronnes dans nn_descriptors.txt...")
            savetxt(PATH + 'nn/nn_descriptors.txt', image_descriptors)
            print("Écriture complétée.")

    else:
        print("Les descripteurs d'images du réseau de neuronnes existent déjà. Chargement des descripteurs.")
        image_descriptors = loadtxt(PATH + 'nn/nn_descriptors.txt')
        image_names = [s.replace('.jpeg', '') for s in os.listdir(PATH + 'jpeg')]

    assert len(os.listdir(PATH + 'jpeg')) == len(image_descriptors)

    ### Chargement des descripteurs de vidéo
    if not os.path.exists(PATH + 'nn/videos'):
        os.makedirs(PATH + 'nn/videos')

    if not len(os.listdir(PATH + 'mp4')) == len(os.listdir(PATH + 'nn/videos')) or re_index:
        print("Calcul des descripteurs des vidéos:")
        start = time.perf_counter()
        video_descriptors = generate_video_descriptors(model)
        end = time.perf_counter()
        print("Calcul des descripteurs des vidéos terminé en", end - start, "secondes.")

        if no_save:
            print("Écritures des descripteurs des vidéos...")
            write_video_descriptors_to_txt(video_descriptors)
            print("Écriture complétée.")
    else:
        print("Les descripteurs de vidéos du réseau de neuronnes existent déjà.")
        video_descriptors = load_video_descriptors(PATH + 'nn/videos')
    
    assert len(os.listdir(PATH + 'mp4')) == len(video_descriptors)

    return image_names, image_descriptors, video_descriptors


def cosine_sim(vector1, vector2) -> float:
    return dot(vector1, vector2) / (norm(vector1) * norm(vector2))


def find_video(image_descriptor, video_descriptors):
    for video_name, descriptors in video_descriptors:
        for i, descriptor in enumerate(descriptors):
            similarity = cosine_sim(descriptor, image_descriptor)

            if similarity >= 0.87:
                return (video_name, i * COMPRESS_RATIO / FPS )
                
    return ('out', '')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store_true', help="Effectue l'indexation peut importe s'il y a déjà des descripteurs")
    parser.add_argument('-n', '--nosave', action='store_false', help="Désactive la sauvegarde des descripteurs")

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
    image_names, image_descriptors, video_descriptors = create_index(args.i, args.nosave)
    end = time.perf_counter()
    print("Phase d'indexation terminée en:", end - start, 'secondes')

    ### Phase de recherche
    print("Début de la phase de recherche")
    start = time.perf_counter()
    with open('nn_answer.csv', 'w') as answers:
        for idx, image_descriptor in enumerate(image_descriptors):
            print("Image:", image_names[idx], end='\r')
            result = find_video(image_descriptor, video_descriptors)
            answers.write('{i},{v},{t}\n'.format(i=image_names[idx], v=result[0], t=result[1]))
            if (idx == 3):
                break
        print()
    end = time.perf_counter()

    search_time = end - start
    print("Recherche terminée en", search_time, "secondes.")
    print("Temps moyen par image:", search_time / len(image_names))
