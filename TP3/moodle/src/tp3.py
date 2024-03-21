import torch
import torchvision
import os
import math
import numpy as np
import time
import cv2 as cv


torch.set_printoptions(precision=1, sci_mode=False)

PATH = '../data/mp4/'
MAX_N_REQUESTS = 1000
FPS = 30
COMPRESS_RATIO = 4
MS_TO_SEC = 1000
LENGTH_HISTOGRAM = 24


def create_index():
    # NxD matrix
    histo_video = []
    
    # Index i, v, t of length N
    index= []
    print('creating index')
    count = 0
    start = time.perf_counter()
    for v, filename in enumerate(os.listdir(PATH)):
        video, _, _ = torchvision.io.read_video(PATH + filename)
        for t in range(0, len(video), COMPRESS_RATIO):
            image = video[t].transpose(0, 2).type(dtype=torch.float32)
            histo_r, _ = torch.histogram(image[0], range=[0, 255], bins=8)
            histo_g, _ = torch.histogram(image[1], range=[0, 255], bins=8)
            histo_b, _ = torch.histogram(image[2], range=[0, 255], bins=8)
            histo_image = torch.cat((histo_r, histo_g, histo_b), 0)
            histo_video.append(histo_image)
            index.append([count, v+1, t])
            count += 1        
    end = time.perf_counter()
    print('index created')
    return index, histo_video, end-start

def read_descriptor():
    db_descriptor = []
    with open('./db_descriptor.txt') as f:
        while True:
            image_descriptor = f.readline()
            if not image_descriptor:
                break
            if image_descriptor != '\n':
                image_descriptor = image_descriptor.rstrip('\r\n').strip()
                image_descriptor = image_descriptor.split(',')
                db_descriptor.append(list(map(float, image_descriptor)))
    return torch.tensor(db_descriptor)

def create_db_descriptor():
    histo_db = []
    indexing_time = 0
    for filename in os.listdir(PATH):
        time, video_desc = create_video_descriptor(PATH+filename) 
        indexing_time += time
        histo_db.append(video_desc)    
    # We take 1 out of every COMPRESSION_RATIO number of frames
    compression_rate = 1 - (1/COMPRESS_RATIO) / 1 
    return indexing_time, compression_rate, histo_db

def len_videos():
    for filename in os.listdir(PATH):
        video, _, _ = torchvision.io.read_video(PATH + filename)
        print(f'{len(video)},')

def create_histogram_single_image(filepath: str):
    image = torchvision.io.read_image(filepath).type(dtype=torch.float32)

    histo_r, _ = torch.histogram(image[0], range=[0, 255], bins=8)
    histo_g, _ = torch.histogram(image[1], range=[0, 255], bins=8)
    histo_b, _ = torch.histogram(image[2], range=[0, 255], bins=8)
    return torch.cat((histo_r, histo_g, histo_b), 0)

def create_requests(n_requests=5):
    assert n_requests <= MAX_N_REQUESTS
    requests = []
    path = '../data/jpeg/'
    for idx, filename in enumerate(os.listdir(path)):
        requests.append(create_histogram_single_image(path + filename))
        if idx == n_requests:
            break
    return requests

def cosine_sim(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    numerator = torch.dot(tensor1, tensor2)
    denominator = torch.sqrt(torch.sum(torch.square(tensor1))) * torch.sqrt(torch.sum(torch.square(tensor2)))
    return numerator/denominator

def euclidean_distance(tensor1: torch.Tensor, tensor2: torch.Tensor):
    return math.sqrt(torch.sum(torch.square(tensor1 - tensor2)))

if __name__ == '__main__':
    requests = create_requests(MAX_N_REQUESTS)
    index, descriptor, indexing_time = create_index() 
    print(f'Indexing time: {indexing_time}')
    
    image_id = [(-1,-1)] * len(requests)
    start = time.perf_counter()
    with open('answer.csv', 'w') as answer_file:
        answer_file.write('image,video_pred,minutage_pred\n')
        for idx, req in enumerate(requests):
            for jdx, histogram in enumerate(descriptor): 
                cosine_similarity = cosine_sim(histogram, req)
                #euclidean_dist = euclidean_distance(histogram, req)
                #if euclidean_dist < 50000:
                #    image_id[i] = idx * COMPRESS_RATIO
                #    break

                if cosine_similarity > 0.97:
                        if index[jdx][1] >= 1:
                            answer_file.write(
                                'i{req:03d},v{id:03d},{min:05f}\n'.format(
                                    req=idx, id=index[jdx][1], min=index[jdx][2]/FPS))
                            break
                elif cosine_similarity <= 0.95 and jdx == (len(descriptor)-1):
                    answer_file.write('i{req:03d},out,\n'.format(req=idx))
                    break
                    
    end = time.perf_counter() 
    print(f'Search time by image: {(end-start)/len(requests)} ms')        
