import numpy as np
import torch
import cv2
import os
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
import torchvision.transforms.transforms as transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True

from data.face_detector import Face_detector
# from data.randaugment import RandAugment
# from data.randaugment import augment_list, Cutout
from data.randaugment import *
from data.randaugment import Cutout
from utils.file_utils import get_img_file, get_vid_file

from PIL import Image
import random

augment_list = [[AutoContrast, 0, 1], [Brightness, 0.35, 0.65], [Color, 0.35, 0.65], [Contrast, 0.35, 0.65],
                [Equalize, 0, 1], [Identity, 0, 1], [Posterize, 4, 8], [Rotate, -10, 10], [Sharpness, 0.35, 0.65], 
                [ShearX, -0.15, 0.15], [ShearY, -0.15, 0.15], [TranslateX, -0.1, 0.1], [TranslateY, -0.1, 0.1]]

# Neutral(0), Anger(1), Disgus(2), Fear(3), Happiness(4), Sadness(5), Surprise(6), Other(7 or -1)
exp_names  = ["Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Other"]

class image_loader(data.dataset.Dataset):
    def __init__(self, data_dir, label_dir, aug_list= "", frame_length=300, audio_path=None, is_s2 = False, is_pred = False, mode="train"):
        self.anno_list = [path for path in os.listdir(label_dir) if '.txt' in path]
        self.frame_length = frame_length
        self.temporal_scale = [0.5, 1.5]
        self.data_dir = data_dir
        self.anno_lens = []
        self.trans_num = 3
        self.mode = mode
        self.labels = []
        
        if aug_list == "":
            self.augment_list = augment_list
        else:
            self.augment_list = aug_list
        
        for txt in self.anno_list:
            with open(os.path.join(label_dir, txt), 'r') as file:
                lines = file.readlines()
            # 각 줄 끝의 개행 문자를 제거.
            lines = [int(line.strip()) if int(line.strip()) != -1 else 7 for line in lines[1:]]
            self.labels.append(lines)
            self.anno_lens.append(len(lines))

        self.total_frame = sum(self.anno_lens)
        self.total_epoch_steps = round(self.total_frame/(frame_length*10))
        self.anno_seq = np.random.randint(len(self.anno_list), size=self.total_epoch_steps)
        print(f"[Dataset] \"{label_dir}\" has", self.total_frame, f"frames, each epoch has {self.total_epoch_steps} steps")
    
    def __len__(self):
        return self.total_epoch_steps

    def __getitem__(self, index):
        while(True):
            file_name = self.anno_list[self.anno_seq[index]]
            label = self.labels[self.anno_seq[index]]
            full_length = self.anno_lens[self.anno_seq[index]]
            
            img_name, direction = get_img_file(self.data_dir[0], file_name)
            # if img_name == "":
            #     img_name, direction = get_img_file(self.data_dir[1], file_name)
            if img_name == "":
                print("[File loading fail]", "directory:", self.data_dir, "file_name:", file_name)
                continue
            
            temporal_scaler = np.random.uniform(low=self.temporal_scale[0], high=self.temporal_scale[1])
            face_vid, label = self.transform(img_name, label, full_length, temporal_scaler, direction=direction)
            
            if np.shape(face_vid) == ():
                index += 1
                if index >= self.__len__():
                    print("[Empty video]", index, self.__len__())
                    print("="*50)
                    index = 0
                continue
                
            # vis_vid = face_vid.cpu().detach().numpy()
            # vis_label = label.cpu().detach().numpy()
            # vis_vid = vis_vid.transpose((1,2,3,0))
            # vis_label = vis_label.squeeze()
            # for img, exp in zip(vis_vid, vis_label):
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #     # 이미지에 텍스트 추가
            #     cv2.putText(img, exp_names[int(exp)], (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            #     cv2.imshow("face_img", img)
            #     key = cv2.waitKey(1)
            
            break
                
        sample = {
            "name": file_name,
            "frames":face_vid,
            "labels": label.type(torch.LongTensor)
        }
        
        return sample
    
    def transform(self, img_dir="", label=[], full_length=300, temporal_scaler = 1.0, direction="front"):
        temporal_scaler = 1.0
        
        if temporal_scaler > 1.0:
            duration = round(self.frame_length*temporal_scaler)
        else:
            duration = self.frame_length
            
        start_vid = round(duration/2)
        end_vid = round(full_length - duration/2) - 1

        if full_length < self.frame_length or start_vid >= end_vid:
            start_point = 1
            end_vid = full_length - 1
            duration = end_vid
        else:
            start_vid = 1
            end_vid = full_length - 1
            start_point = np.random.randint(low=start_vid, high=end_vid)
            if start_point + duration > full_length:
                duration = full_length - start_point
            
        face_vid = []
        
        if self.augment_list:
            ops = random.choices(self.augment_list, k=self.trans_num)
        else:
            ops = []
        vals = []
        for _, min_val, max_val in ops:
            val = min_val + float(max_val - min_val)*random.random()
            vals.append(val)
            
        for idx in range(duration-1):
            curr_idx = start_point + idx
            filename = f"{curr_idx:05d}.jpg"
            cropped_face = cv2.imread(os.path.join(img_dir, filename))
                
            try:
                cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            except:
                cropped_face = np.zeros((256,256,3), dtype=np.uint8)
                label[curr_idx] = 7
                
            cropped_face = cv2.resize(cropped_face, dsize=(256,256))
            cropped_face = Image.fromarray(cropped_face)
            
            if self.mode == "train":
                for (op, _, _), val in zip(ops, vals):
                    op = eval(op)
                    cropped_face = op(cropped_face, val)
                if random.random() > 0.5:
                    cutout_val = random.random()/2.0 # [0, 0.5]
                    cropped_face = Cutout(cropped_face, cutout_val) #for fixmatch
                
            face_vid.append(np.asarray(cropped_face, dtype=np.uint8))
        
        face_vid = np.asarray(face_vid, dtype=np.float32)
        face_vid /= 255.0
                
        resampler = torch.nn.Upsample(size=(self.frame_length, 256, 256), mode='nearest')
        face_vid = face_vid.transpose((3,0,1,2))
        face_vid = torch.from_numpy(face_vid)
        face_vid = resampler(face_vid.unsqueeze(0))[0]
        
        resampler = torch.nn.Upsample(size=(self.frame_length,), mode='nearest')
        label = label[start_point:start_point+duration]
        label = torch.FloatTensor(label)
        label = resampler(label.view(1, 1, -1))

        return face_vid, label
    
class VIG_dataloader(data.dataset.Dataset):
    def __init__(self, data_dir, label_dir, aug_list= "", frame_length=300, audio_path=None, is_s2 = False, is_pred = False, mode="train"):
        self.data_dir = data_dir
        self.anno_list = [path for path in os.listdir(label_dir) if '.txt' in path]
        self.labels = []
        self.anno_lens = []
        self.mode = mode
        
        self.frame_length = frame_length
        self.temporal_scale = [0.7, 1.3]
        self.detector = Face_detector()
        self.trans_num = 3
        
        if aug_list == "":
            self.augment_list = [
                [AutoContrast, 0, 1],
                [Brightness, 0.35, 0.65],
                [Color, 0.35, 0.65],
                [Contrast, 0.35, 0.65],
                [Equalize, 0, 1],
                [Identity, 0, 1],
                [Posterize, 4, 8],
                [Rotate, -10, 10],
                [Sharpness, 0.35, 0.65],
                [ShearX, -0.15, 0.15],
                [ShearY, -0.15, 0.15],
                [TranslateX, -0.1, 0.1],
                [TranslateY, -0.1, 0.1]
            ]
        else:
            self.augment_list = aug_list
            
        self.face_img = np.zeros((256,256,3), dtype=np.uint8)
        
        for txt in self.anno_list:
            with open(os.path.join(label_dir, txt), 'r') as file:
                lines = file.readlines()
            # 각 줄 끝의 개행 문자를 제거합니다.
            lines = [int(line.strip()) if int(line.strip()) != -1 else 7 for line in lines[1:]]
            self.labels.append(lines)
            self.anno_lens.append(len(lines))

        self.total_frame = sum(self.anno_lens)
        self.epoch_steps = round(self.total_frame/frame_length)
        self.anno_seq = np.random.randint(len(self.anno_list), size=self.epoch_steps)
        print(f"[Dataset] \"{label_dir}\" has", self.total_frame, f"frames, each epoch has {self.epoch_steps} steps")
    
    def __len__(self):
        return self.epoch_steps

    def __getitem__(self, index):
        
        while(True):
            file_name = self.anno_list[self.anno_seq[index]]
            label = self.labels[self.anno_seq[index]]
            full_length = self.anno_lens[self.anno_seq[index]]

            vid_name, direction = get_vid_file(self.data_dir, file_name)
            
            if vid_name == "":
                print("[File loading fail]", "directory:", self.data_dir, "file_name:", file_name)
                continue
            
            temporal_scaler = np.random.uniform(low=self.temporal_scale[0], high=self.temporal_scale[1])
            face_vid, label = self.transform(vid_name, label, full_length, temporal_scaler, direction=direction)
            
            # # dummy
            # face_vid = torch.rand(3, 100, 256, 256)
            # resampler = torch.nn.Upsample(size=(self.frame_length,), mode='nearest')
            # label = label[:100]
            # label = torch.FloatTensor(label)
            # label = resampler(label.view(1, 1, -1))
            # label = torch.FloatTensor(label)
            
            if np.shape(face_vid) == ():
                index += 1
                if index >= self.__len__():
                    print("[Empty video]", index, self.__len__())
                    print("="*50)
                    index = 0
                continue
                
            # vis_vid = face_vid.cpu().detach().numpy()
            # vis_label = label.cpu().detach().numpy()
            # vis_vid = vis_vid.transpose((1,2,3,0))
            # vis_label = vis_label.squeeze()
            # for img, exp in zip(vis_vid, vis_label):
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #     # 이미지에 텍스트 추가
            #     cv2.putText(img, exp_names[int(exp)], (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            #     cv2.imshow("face_img", img)
            #     key = cv2.waitKey(1)
                
            break
                
        sample = {
            "name": file_name,
            "frames":face_vid,
            "labels": label.type(torch.LongTensor)
        }
        
        return sample
    
    def transform(self, vid_name="", label=[], full_length=300, temporal_scaler = 1.0, direction="front"):
        
        if temporal_scaler > 1.0:
            duration = round(self.frame_length*temporal_scaler)
        else:
            duration = self.frame_length
        
        start_vid = round(duration/2)
        end_vid = round(full_length - duration/2) - 1

        if full_length < self.frame_length or start_vid >= end_vid:
            start_vid = 1
            end_vid = full_length - 1

        start_point = np.random.randint(low=start_vid, high=end_vid)
        start_point = 0
        
        self.detector.reset()
        cap = cv2.VideoCapture(vid_name)
        # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_point)

        face_vid = []
        
        if self.augment_list:
            ops = random.choices(self.augment_list, k=self.trans_num)
        else:
            ops = []
            
        cutout_val = random.random() * 0.5 

        vals = []
        for _, min_val, max_val in ops:
            val = min_val + float(max_val - min_val)*random.random()
            vals.append(val)
        
        for i in range(duration-1):
            res, frame = cap.read()
            curr_idx = start_point + i

            if not res:
                if face_vid:
                    continue
                else:
                    print("[video load fail] vid_name", vid_name,"sp", start_point,"curr_idx", curr_idx)
                    return -1, -1
            
            cropped_face = self.detector.detection(frame, direction=direction)
            
            try:
                cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            except:
                cropped_face = np.zeros((256,256,3), dtype=np.uint8)
                label[curr_idx] = 7
                
            cropped_face = cv2.resize(cropped_face, dsize=(256,256))
            cropped_face = Image.fromarray(cropped_face)
            
            if self.mode == "train":
                for (op, _, _), val in zip(ops, vals):
                    op = eval(op)
                    cropped_face = op(cropped_face, val) 
                cropped_face = Cutout(cropped_face, cutout_val) #for fixmatch
                
            face_vid.append(np.asarray(cropped_face, dtype=np.uint8))
        
        face_vid = np.asarray(face_vid, dtype=np.float32)
        face_vid /= 255.0
                
        resampler = torch.nn.Upsample(size=(self.frame_length, 256, 256), mode='nearest')
        face_vid = face_vid.transpose((3,0,1,2))
        face_vid = torch.from_numpy(face_vid)
        face_vid = resampler(face_vid.unsqueeze(0))[0]
        
        resampler = torch.nn.Upsample(size=(self.frame_length,), mode='nearest')
        label = label[start_point:start_point+duration]
        label = torch.FloatTensor(label)
        label = resampler(label.view(1, 1, -1))

        return face_vid, label
        
def build_seq_dataset(config, mode):
    if not config.use_audio_fea: 
        if mode == "train":
            print("Dataloader building... [Train set]")
            dataset = image_loader(data_dir=config.img_data_path, label_dir=config.train_anno_path, aug_list=config.train_aug_list, frame_length=config.frame_length, mode="train")
            # dataset = VIG_dataloader(data_dir=config["train_vid_path"], label_dir=config["train_anno_path"], aug_list=config["train_aug_list"], frame_length=config["frame_length"], mode="train")
        elif mode =="val":
            print("Dataloader building... [Val set]")
            dataset = image_loader(data_dir=config.img_data_path, label_dir=config.val_anno_path, aug_list=config.train_aug_list, frame_length=config.frame_length, mode="train")
            # dataset = VIG_dataloader(data_dir=config["val_vid_path"], label_dir=config["val_anno_path"], aug_list=config["val_aug_list"], frame_length=config["frame_length"], mode="val")
            
            
    return dataset


if __name__ == '__main__':
    # train_transform = build_transform(True)
    
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.ToTensor()])

    # ABAW5Seqdata(config["train_img_path"], transform)
    dataset = VIG_dataloader("../../cropped_data/", "../../EXPR_Classification_Challenge/Train_Set", transform=transform)
    data = dataset.__getitem__(0)


