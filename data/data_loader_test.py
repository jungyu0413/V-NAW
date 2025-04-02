import numpy as np
import torch
import cv2
import os
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True
from misc import *
from data.randaugment import *
from data.randaugment import Cutout
from PIL import Image
import random
import copy

# Neutral(0), Anger(1), Disgus(2), Fear(3), Happiness(4), Sadness(5), Surprise(6), Other(7 or -1)
exp_names  = ["Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Other"]

class image_loader(data.dataset.Dataset):
    def __init__(self, args, data_dir, label_dir, frame_length=32, mode="train", task="exp"):
        
        video_files = []
        with open(label_dir, 'r') as file:
            lines = file.readlines()
            video_files = [line.strip() for line in lines]
            
        self.video_files = []
        
        for file in video_files:
            vid_path, _ = get_img_file(data_dir, file, 0.0, "valid")
            self.video_files.append(vid_path)
        
        self.data_list = []
        
        for file in self.video_files:
            img_files = os.listdir(file)
            fns = [int(frame.split(".")[0]) for frame in img_files if "jpg" in frame]
            last_img_idx = max(fns)
            # print(file, ":", last_img_idx)
            # last_img_idx = int(img_files[-1].split(".")[0]) - 100
            
            for i in range(1, last_img_idx+1):
                sp = i
                ep = i + (frame_length-1)
                self.data_list.append([file, sp, ep])
            
        self.frame_length = frame_length

        self.mean = np.array([0.577, 0.4494, 0.4001], dtype=np.float32).reshape((-1, 1, 1, 1))  # match dimension
        self.std = np.array([0.2628, 0.2395, 0.2383], dtype=np.float32).reshape((-1, 1, 1, 1))
        self.mean = torch.from_numpy(self.mean)
        self.std  = torch.from_numpy(self.std)
        
        self.data_dir = data_dir
        self.mode = mode
        self.task = task

        self.total_num= len(self.data_list)
        print(f"[Dataset] \"{data_dir}\" has", self.total_num, "videos")
        
    def __len__(self):
        return self.total_num

    def __getitem__(self, index):
        while True:
            file, sp, ep = self.data_list[index]
            face_vid, target_frame = self.transform(file, sp, ep)

            if np.shape(face_vid) == ():
                index += 1
                if index >= self.__len__():
                    print("[Empty video]", index, self.__len__())
                    print("="*50)
                    index = 0
                continue
            
            break

        return face_vid, target_frame


    def transform(self, img_dir="", sp=0, ep=100):
        
        face_vid = []
        empty_queue = [0] * 100
        video_name = img_dir.split("/")[-1]
        target_frame = f"{video_name}/{sp:05d}.jpg"
        
        for num, curr_idx in enumerate(range(sp, ep+1)):
            filename = f"{curr_idx:05d}.jpg"
            file_path = os.path.join(img_dir, filename)
            if os.path.isfile(file_path):
                cropped_face = cv2.imread(os.path.join(img_dir, filename))
            else:
                cropped_face = np.zeros((112,112,3), dtype=np.uint8)
                empty_queue[num] = 1
            
            try:
                cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                cropped_face = cv2.resize(cropped_face, dsize=(112,112))
                cropped_face = Image.fromarray(cropped_face)
            except:
                cropped_face = np.zeros((112,112,3), dtype=np.uint8)
                cropped_face = Image.fromarray(cropped_face)
                # print('img load error:', img_dir, idx)
                empty_queue[curr_idx] = 1
                
            face_vid.append(np.asarray(cropped_face, dtype=np.uint8))
            
        if sum(empty_queue) == len(empty_queue):
            return -1, -1
        
        temp_que = copy.deepcopy(empty_queue)
        temp_que = np.array(empty_queue)
        # 1의 위치 찾기
        one_indices = np.where(temp_que == 1)[0]
        zero_indices = np.where(temp_que == 0)[0]
        
        for empty_idx in one_indices:
            distance = np.abs(zero_indices.copy() - empty_idx)
            padding_idx = distance.argmin()
            try:
                face_vid[empty_idx] = face_vid[zero_indices[padding_idx]].copy()
            except:
                
                print("padding_idx",padding_idx)
                print("empty_idx",empty_idx)
                print("face_vid",np.shape(face_vid))
                
                print("padding_idx",padding_idx)
                print("empty_idx",empty_idx)
                print("face_vid",np.shape(face_vid))
                exit(0)
        
        face_vid = np.asarray(face_vid, dtype=np.float32)
        face_vid /= 255.0
                
        resampler = torch.nn.Upsample(size=(self.frame_length, 112, 112), mode='nearest')
        face_vid = face_vid.transpose((3,0,1,2))
        face_vid = torch.from_numpy(face_vid)
        face_vid = (face_vid - self.mean)/self.std
        face_vid = resampler(face_vid.unsqueeze(0))[0]
        face_vid = torch.transpose(face_vid, 0, 1)
        
        return face_vid, target_frame
        

def build_seq_dataset(args, mode="train", task="exp"):
    print("Dataloader building... [Test set]")
    dataset = image_loader(args, data_dir=args.data_dir, label_dir=args.test_vid_path, \
        frame_length=args.clip, mode="valid", task=task)
            
    return dataset

# if __name__ == '__main__':
#     dataset = image_loader(data_dir=r"C:\Users\user\Desktop\NLA\ABAW8", label_dir=r"C:\Users\user\Desktop\NLA\ABAW8\EXPR_Recognition_Challenge\Train_Set", frame_length=100, mode="train", task='exp')
#     data = dataset.__getitem__(0)