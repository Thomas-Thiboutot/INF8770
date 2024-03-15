import torchvision
import torch
import os


torch.set_printoptions(precision=1, sci_mode=False)    

PATH = '../data/mp4/'


def create_video_descriptor(filename: str):
    video, _, _ = torchvision.io.read_video(filename)
    histo_video = []
    for image in video: 
        h = image.transpose(0, 2).type(dtype=torch.float32)
        histo_r, _ = torch.histogram(h[0], range=[0,255], bins = 8)
        histo_g, _ = torch.histogram(h[1], range=[0,255], bins = 8)
        histo_b, _ = torch.histogram(h[2], range=[0,255], bins = 8)
        histo_image = torch.cat((histo_r, histo_g, histo_b), 0)
        # print(','.join(map(str, histo_image.tolist())), '\n')
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
    return db_descriptor
        
def create_db_descriptor():
    histo_db = []
    for file in os.listdir(PATH):
        histo_db.append(create_video_descriptor(PATH+file))
    return histo_db
        

if __name__ == '__main__':
    db_descriptor = read_descriptor()
    print(db_descriptor[0])
    