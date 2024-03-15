import torchvision
import torch
import os
import math


torch.set_printoptions(precision=1, sci_mode=False)

PATH = '../data/mp4/'
MAX_N_REQUESTS = 1000
FPS = 30


def create_video_descriptor(filename: str):
    video, _, _ = torchvision.io.read_video(filename)
    histo_video = []
    for image in video:
        image = image.transpose(0, 2).type(dtype=torch.float32)
        histo_r, _ = torch.histogram(image[0], range=[0, 255], bins=8)
        histo_g, _ = torch.histogram(image[1], range=[0, 255], bins=8)
        histo_b, _ = torch.histogram(image[2], range=[0, 255], bins=8)
        histo_image = torch.cat((histo_r, histo_g, histo_b), 0)
        histo_video.append(histo_image)
    return histo_video


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
    for filename in os.listdir(PATH):
        histo_db.append(create_video_descriptor(PATH + filename))
    return histo_db


def len_videos():
    for filename in os.listdir(PATH):
        video, _, _ = torchvision.io.read_video(PATH + filename)
        print(f'{len(video)},')


def create_dict_videos():
    videos_length = {}
    video_id = 0
    with open('./len_video.txt') as len_list:
        while True:
            video_id += 1
            image_descriptor = len_list.readline()
            if not image_descriptor:
                break
            if image_descriptor != '\n':
                image_descriptor = image_descriptor.rstrip('\r\n').strip()
                image_descriptor = image_descriptor.split(',')
                videos_length[video_id] = int(image_descriptor[0])
    return videos_length


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


if __name__ == '__main__':
    descriptor = read_descriptor()
    length_videos = create_dict_videos()
    print('Initialize requests')
    requests = create_requests()
    print('request ready')
    path = '../data/jpeg/'
    image_id = [-1] * len(requests)
    for i, req in enumerate(requests):
        for idx, histogram in enumerate(descriptor):
            euclidean_dist = math.sqrt(
                torch.sum(torch.square(histogram - req)))
            if euclidean_dist < 35000:
                image_id[i] = idx
                break
    with open('answer.txt', 'w') as answer_file:
        answer_file.write('image,video_pred,minutage_pred\n')
        for j, im_id in enumerate(image_id):
            time = 0
            minuting = -1
            for video_id, n_image in length_videos.items():
                time += n_image
                if im_id <= time:
                    minuting = (n_image - (time - im_id)) / FPS
                    if minuting > 0:
                        answer_file.write(
                            'i{req:03d},v{id:03d},{min:05f}\n'.format(
                                req=j, id=video_id, min=minuting))
                    else:
                        answer_file.write('i{req:03d},out,\n'.format(req=j))
                    break
