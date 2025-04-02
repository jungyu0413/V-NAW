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
from torchsampler import ImbalancedDatasetSampler




# Neutral(0), Anger(1), Disgus(2), Fear(3), Happiness(4), Sadness(5), Surprise(6), Other(7 or -1)
exp_names  = ["Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Other"]

class image_loader(data.dataset.Dataset):
    def __init__(self, args, data_dir, label_dir, frame_length=32, mode="train", task="exp"):
        self.anno_list = [path for path in os.listdir(label_dir) if '.txt' in path]
        self.frame_length = frame_length
        self.trans_num = args.tras_n
        if args.temporal_aug:
            self.temporal_scale = [0.6, 1.4]
        else:
            self.temporal_scale = []
        self.mean = np.array([0.577, 0.4494, 0.4001], dtype=np.float32).reshape((-1, 1, 1, 1))  # match dimension
        self.std = np.array([0.2628, 0.2395, 0.2383], dtype=np.float32).reshape((-1, 1, 1, 1))
        self.mean = torch.from_numpy(self.mean)
        self.std  = torch.from_numpy(self.std)
        
        self.data_dir = data_dir
        self.anno_lens = []
        self.mode = mode
        self.task = task
        self.labels = []
        if self.trans_num == 0:
            self.augment_list = [["Identity", 0, 1],["Identity", 0, 1],["Identity", 0, 1]]
            self.trans_num = 1
        else:
            aug_params = [0.6, 1.4]
            # self.augment_list =  [["AutoContrast", aug_params[0], aug_params[1]], ["Brightness", aug_params[0], aug_params[1]], \
            #     ["Color", aug_params[0], aug_params[1]], ["Contrast", aug_params[0], aug_params[1]], \
            #     ["Equalize", aug_params[0], aug_params[1]], ["Identity", 0, 1], ["Identity", 0, 1], ["Identity", 0, 1], ["Identity", 0, 1], 
            #     ["Identity", 0, 1], ["Identity", 0, 1], ["Posterize", 6, 8], ["Rotate", -10, 10], ["Sharpness", aug_params[0], aug_params[1]], \
            #     ["ShearX", -0.1, 0.1], ["ShearY", -0.1, 0.1], ["TranslateX", -0.1, 0.1], ["TranslateY", -0.1, 0.1]]
            self.augment_list =  [["AutoContrast", aug_params[0], aug_params[1]], ["Brightness", aug_params[0], aug_params[1]], \
                ["Color", aug_params[0], aug_params[1]], ["Contrast", aug_params[0], aug_params[1]], \
                ["Equalize", aug_params[0], aug_params[1]], ["Identity", 0, 1], ["Identity", 0, 1],
                ["Posterize", 6, 8], ["Sharpness", aug_params[0], aug_params[1]]]
            
        for txt in self.anno_list:
            with open(os.path.join(label_dir, txt), 'r') as file:
                lines = file.readlines()
            # 각 줄 끝의 개행 문자를 제거.
            if self.task == "exp":
                lines = [int(line.strip()) if int(line.strip()) != -1 else 7 for line in lines[1:]]
            elif self.task == "va":
                lines = [[float(line.strip().split(',')[0]), float(line.strip().split(',')[1])] for line in lines[1:]]
            elif self.task == "au":
                lines = [[int(item) for item in line.strip().split(',')] for line in lines[1:]]
                
            self.labels.append(lines)
            self.anno_lens.append(len(lines))

        self.total_frame = sum(self.anno_lens)
        
        if self.mode == "train":
            self.epoch_steps = len(self.anno_list) 
            # self.epoch_steps = len(self.anno_list)*3
            self.anno_seq = np.random.randint(len(self.anno_list), size=self.epoch_steps)
            print(f"[Dataset] \"{label_dir}\" has", self.total_frame, f"frames, each epoch has {self.epoch_steps} steps")
        else:
            # self.epoch_steps = len(self.anno_list)
            self.epoch_steps = round(self.total_frame/(frame_length*30))
            self.anno_seq = np.random.randint(len(self.anno_list), size=self.epoch_steps)
            print(f"[Dataset] \"{label_dir}\" has", self.total_frame, f"frames, each epoch has {self.epoch_steps} steps")
    
    def __len__(self):
        return self.epoch_steps

    def __getitem__(self, index):
        while(True):
            if self.mode == "train":
                file_name = self.anno_list[self.anno_seq[index]]
                label = self.labels[self.anno_seq[index]]
                full_length = len(label)
            else:
                file_name = self.anno_list[self.anno_seq[index]]
                label = self.labels[self.anno_seq[index]]
                full_length = self.anno_lens[self.anno_seq[index]]
                
            img_name, _ = get_img_file(self.data_dir, file_name, random.random(), self.mode)
            
            # print("[DATA_]", len(os.listdir(img_name)), len(label), "full", full_length)
            
            if img_name == "":
                # print("[File loading fail]", "directory:", self.data_dir, "file_name:", file_name)
                continue
            
            if self.temporal_scale and self.mode == "train":
                temporal_scaler = np.random.uniform(low=self.temporal_scale[0], high=self.temporal_scale[1])
                face_vid, flipped_vid, label = self.transform(img_name, label, full_length, temporal_scaler)
            else:
                face_vid, flipped_vid, label = self.transform(img_name, label, full_length)
            
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
                
        # sample = {
        #     "name": file_name,
        #     "frames":face_vid,
        #     "labels": label.type(torch.LongTensor)
        # }
        if self.mode == "train":
            return face_vid, flipped_vid, label
        else:
            return face_vid, label
        


    def transform(self, img_dir="", label=[], full_length=300, temporal_scaler = 1.0):
        
        duration = round(self.frame_length*temporal_scaler)
            
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
        flipped_vid = []
        
        ops = random.choices(self.augment_list, k=self.trans_num)
        
        vals = []
        for _, min_val, max_val in ops:
            val = min_val + float(max_val - min_val)*random.random()
            vals.append(val)
            
        empty_queue = [0] * (duration-1)
        
        for idx in range(duration-1):
            curr_idx = start_point + idx
            filename = f"{curr_idx:05d}.jpg"
            file_path = os.path.join(img_dir, filename)
            if os.path.isfile(file_path):
                cropped_face = cv2.imread(os.path.join(img_dir, filename))
            else:
                cropped_face = np.zeros((112,112,3), dtype=np.uint8)
                empty_queue[idx] = 1
                # if self.task == 'exp':
                #     label[curr_idx] = 7
                # elif self.task == "va":
                #     label[curr_idx] = [0.0, 0.0]
                # elif self.task == "au":
                #     label[curr_idx] = [0]*12
            
            try:
                cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                cropped_face = cv2.resize(cropped_face, dsize=(112,112))
                cropped_face = Image.fromarray(cropped_face)
            except:
                cropped_face = np.zeros((112,112,3), dtype=np.uint8)
                cropped_face = Image.fromarray(cropped_face)
                # print('img load error:', img_dir, idx)
                empty_queue[idx] = 1
                
            if self.mode == "train":
                for (op, _, _), val in zip(ops, vals):
                    op = eval(op)
                    cropped_face = op(cropped_face, val)
                if random.random() > 0.5:
                    cutout_val = random.random()/2.0 # [0, 0.5]
                    cropped_face = Cutout(cropped_face, cutout_val) #for fixmatch
                
            face_vid.append(np.asarray(cropped_face, dtype=np.uint8))
            flipped_vid.append(cv2.flip(np.asarray(cropped_face, dtype=np.uint8), 1))
            # cv2.imwrite('face_vid.jpg', np.asarray(cropped_face, dtype=np.uint8))
            # cv2.imwrite('flipped_vid.jpg', cv2.flip(np.asarray(cropped_face, dtype=np.uint8), 1))
            
        if sum(empty_queue) == len(empty_queue):
            return -1, -1, -1
        
        temp_que = copy.deepcopy(empty_queue)
        temp_que = np.array(empty_queue)
        # 1의 위치 찾기
        one_indices = np.where(temp_que == 1)[0]
        zero_indices = np.where(temp_que == 0)[0]
        
        label = label[start_point:start_point+duration]
        
        for empty_idx in one_indices:
            distance = np.abs(zero_indices.copy() - empty_idx)
            padding_idx = distance.argmin()
            face_vid[empty_idx] = face_vid[zero_indices[padding_idx]].copy()
            flipped_vid[empty_idx] = flipped_vid[zero_indices[padding_idx]].copy()
            label[empty_idx] = label[zero_indices[padding_idx]]
        
        face_vid = np.asarray(face_vid, dtype=np.float32)
        face_vid /= 255.0
        flipped_vid = np.asarray(flipped_vid, dtype=np.float32)
        flipped_vid /= 255.0
                
        resampler = torch.nn.Upsample(size=(self.frame_length, 112, 112), mode='nearest')
        face_vid = face_vid.transpose((3,0,1,2))
        face_vid = torch.from_numpy(face_vid)
        face_vid = (face_vid - self.mean)/self.std
        face_vid = resampler(face_vid.unsqueeze(0))[0]
        face_vid = torch.transpose(face_vid, 0, 1)
        
        flipped_vid = flipped_vid.transpose((3,0,1,2))
        flipped_vid = torch.from_numpy(flipped_vid)
        flipped_vid = (flipped_vid - self.mean)/self.std
        flipped_vid = resampler(flipped_vid.unsqueeze(0))[0]
        flipped_vid = torch.transpose(flipped_vid, 0, 1)
        
        resampler = torch.nn.Upsample(size=(self.frame_length,), mode='nearest')
        # label = label[start_point:start_point+duration]
        
        if self.task == 'exp':
            label = torch.FloatTensor(label)
            label = resampler(label.view(1, 1, -1))
            label = label.long()
        elif self.task == 'va':
            label = np.asarray(label, dtype=np.float32)
            label = torch.FloatTensor(label)
            label = resampler(label.view(1, 2, -1))
        elif self.task == 'au':
            label = np.asarray(label, dtype=np.float32)
            label = torch.FloatTensor(label)
            label = resampler(label.view(1, 12, -1))
            label = label.long()

        return face_vid, flipped_vid, label
        




random_erasing = transforms.RandomErasing(p=1.0, scale=(0.02, 0.3))

def apply_random_erasing(image):
    """
    NumPy 이미지를 PyTorch 텐서로 변환 후 RandomErasing 적용하고 다시 NumPy로 변환
    """
    # NumPy 배열 (H, W, C) → PyTorch 텐서 (C, H, W)
    image_tensor = transforms.ToTensor()(image)

    # RandomErasing 적용
    erased_tensor = random_erasing(image_tensor)

    # 다시 NumPy 형식으로 변환 (C, H, W) → (H, W, C)
    erased_image = transforms.ToPILImage()(erased_tensor)
    return np.asarray(erased_image, dtype=np.uint8)



class image_loader_erasing(data.dataset.Dataset):
    def __init__(self, args, data_dir, label_dir, frame_length=32, mode="train", task="exp"):
        self.anno_list = [path for path in os.listdir(label_dir) if '.txt' in path]
        self.frame_length = frame_length
        self.trans_num = args.tras_n
        if args.temporal_aug:
            self.temporal_scale = [0.6, 1.4]
        else:
            self.temporal_scale = []
        self.mean = np.array([0.577, 0.4494, 0.4001], dtype=np.float32).reshape((-1, 1, 1, 1))  # match dimension
        self.std = np.array([0.2628, 0.2395, 0.2383], dtype=np.float32).reshape((-1, 1, 1, 1))
        self.mean = torch.from_numpy(self.mean)
        self.std  = torch.from_numpy(self.std)
        
        self.data_dir = data_dir
        self.anno_lens = []
        self.mode = mode
        self.task = task
        self.labels = []
        if self.trans_num == 0:
            self.augment_list = [["Identity", 0, 1],["Identity", 0, 1],["Identity", 0, 1]]
            self.trans_num = 1
        else:
            aug_params = [0.6, 1.4]
            # self.augment_list =  [["AutoContrast", aug_params[0], aug_params[1]], ["Brightness", aug_params[0], aug_params[1]], \
            #     ["Color", aug_params[0], aug_params[1]], ["Contrast", aug_params[0], aug_params[1]], \
            #     ["Equalize", aug_params[0], aug_params[1]], ["Identity", 0, 1], ["Identity", 0, 1], ["Identity", 0, 1], ["Identity", 0, 1], 
            #     ["Identity", 0, 1], ["Identity", 0, 1], ["Posterize", 6, 8], ["Rotate", -10, 10], ["Sharpness", aug_params[0], aug_params[1]], \
            #     ["ShearX", -0.1, 0.1], ["ShearY", -0.1, 0.1], ["TranslateX", -0.1, 0.1], ["TranslateY", -0.1, 0.1]]
            self.augment_list =  [["AutoContrast", aug_params[0], aug_params[1]], ["Brightness", aug_params[0], aug_params[1]], \
                ["Color", aug_params[0], aug_params[1]], ["Contrast", aug_params[0], aug_params[1]], \
                ["Equalize", aug_params[0], aug_params[1]], ["Identity", 0, 1], ["Identity", 0, 1],
                ["Posterize", 6, 8], ["Sharpness", aug_params[0], aug_params[1]]]
            
        for txt in self.anno_list:
            with open(os.path.join(label_dir, txt), 'r') as file:
                lines = file.readlines()
            # 각 줄 끝의 개행 문자를 제거.
            if self.task == "exp":
                lines = [int(line.strip()) if int(line.strip()) != -1 else 7 for line in lines[1:]]
            elif self.task == "va":
                lines = [[float(line.strip().split(',')[0]), float(line.strip().split(',')[1])] for line in lines[1:]]
            elif self.task == "au":
                lines = [[int(item) for item in line.strip().split(',')] for line in lines[1:]]
                
            self.labels.append(lines)
            self.anno_lens.append(len(lines))

        self.total_frame = sum(self.anno_lens)
        
        if self.mode == "train":
            self.epoch_steps = len(self.anno_list) 
            # self.epoch_steps = len(self.anno_list)*3
            self.anno_seq = np.random.randint(len(self.anno_list), size=self.epoch_steps)
            print(f"[Dataset] \"{label_dir}\" has", self.total_frame, f"frames, each epoch has {self.epoch_steps} steps")
        else:
            # self.epoch_steps = len(self.anno_list)
            self.epoch_steps = round(self.total_frame/(frame_length*30))
            self.anno_seq = np.random.randint(len(self.anno_list), size=self.epoch_steps)
            print(f"[Dataset] \"{label_dir}\" has", self.total_frame, f"frames, each epoch has {self.epoch_steps} steps")
    
    def __len__(self):
        return self.epoch_steps

    def __getitem__(self, index):
        while(True):
            if self.mode == "train":
                file_name = self.anno_list[self.anno_seq[index]]
                label = self.labels[self.anno_seq[index]]
                full_length = len(label)
            else:
                file_name = self.anno_list[self.anno_seq[index]]
                label = self.labels[self.anno_seq[index]]
                full_length = self.anno_lens[self.anno_seq[index]]
                
            img_name, _ = get_img_file(self.data_dir, file_name, random.random(), self.mode)
            
            # print("[DATA_]", len(os.listdir(img_name)), len(label), "full", full_length)
            
            if img_name == "":
                print("[File loading fail]", "directory:", self.data_dir, "file_name:", file_name)
                continue
            
            if self.temporal_scale and self.mode == "train":
                temporal_scaler = np.random.uniform(low=self.temporal_scale[0], high=self.temporal_scale[1])
                face_vid, flipped_vid, label = self.transform(img_name, label, full_length, temporal_scaler)
            else:
                face_vid, flipped_vid, label = self.transform(img_name, label, full_length)
            
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
                
        # sample = {
        #     "name": file_name,
        #     "frames":face_vid,
        #     "labels": label.type(torch.LongTensor)
        # }
        if self.mode == "train":
            return face_vid, flipped_vid, label
        else:
            return face_vid, label
        


    def transform(self, img_dir="", label=[], full_length=300, temporal_scaler = 1.0):
        
        duration = round(self.frame_length*temporal_scaler)
            
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
        flipped_vid = []
        
        ops = random.choices(self.augment_list, k=self.trans_num)
        
        vals = []
        for _, min_val, max_val in ops:
            val = min_val + float(max_val - min_val)*random.random()
            vals.append(val)
            
        empty_queue = [0] * (duration-1)
        
        for idx in range(duration-1):
            curr_idx = start_point + idx
            filename = f"{curr_idx:05d}.jpg"
            file_path = os.path.join(img_dir, filename)
            if os.path.isfile(file_path):
                cropped_face = cv2.imread(os.path.join(img_dir, filename))
            else:
                cropped_face = np.zeros((112,112,3), dtype=np.uint8)
                empty_queue[idx] = 1
                # if self.task == 'exp':
                #     label[curr_idx] = 7
                # elif self.task == "va":
                #     label[curr_idx] = [0.0, 0.0]
                # elif self.task == "au":
                #     label[curr_idx] = [0]*12
            
            try:
                cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                cropped_face = cv2.resize(cropped_face, dsize=(112,112))
                cropped_face = Image.fromarray(cropped_face)
            except:
                cropped_face = np.zeros((112,112,3), dtype=np.uint8)
                cropped_face = Image.fromarray(cropped_face)
                # print('img load error:', img_dir, idx)
                empty_queue[idx] = 1
                
            if self.mode == "train":
                for (op, _, _), val in zip(ops, vals):
                    op = eval(op)
                    cropped_face = op(cropped_face, val)
                if random.random() > 0.5:
                    cutout_val = random.random()/2.0 # [0, 0.5]
                    cropped_face = Cutout(cropped_face, cutout_val) #for fixmatch
                
            #face_vid.append(apply_random_erasing(np.asarray(cropped_face, dtype=np.uint8)))
            #flipped_vid.append(apply_random_erasing(cv2.flip(np.asarray(cropped_face, dtype=np.uint8), 1)))
            face_vid.append(np.asarray(cropped_face, dtype=np.uint8))
            flipped_vid.append(cv2.flip(np.asarray(cropped_face, dtype=np.uint8), 1))
            # cv2.imwrite('face_vid.jpg', np.asarray(cropped_face, dtype=np.uint8))
            # cv2.imwrite('flipped_vid.jpg', cv2.flip(np.asarray(cropped_face, dtype=np.uint8), 1))
            
        if sum(empty_queue) == len(empty_queue):
            return -1, -1, -1
        
        temp_que = copy.deepcopy(empty_queue)
        temp_que = np.array(empty_queue)
        # 1의 위치 찾기
        one_indices = np.where(temp_que == 1)[0]
        zero_indices = np.where(temp_que == 0)[0]
        
        label = label[start_point:start_point+duration]
        
        for empty_idx in one_indices:
            distance = np.abs(zero_indices.copy() - empty_idx)
            padding_idx = distance.argmin()
            face_vid[empty_idx] = face_vid[zero_indices[padding_idx]].copy()
            flipped_vid[empty_idx] = flipped_vid[zero_indices[padding_idx]].copy()
            label[empty_idx] = label[zero_indices[padding_idx]]
        
        face_vid = np.asarray(face_vid, dtype=np.float32)
        face_vid /= 255.0
        flipped_vid = np.asarray(flipped_vid, dtype=np.float32)
        flipped_vid /= 255.0
                
        resampler = torch.nn.Upsample(size=(self.frame_length, 112, 112), mode='nearest')
        face_vid = face_vid.transpose((3,0,1,2))
        face_vid = torch.from_numpy(face_vid)
        face_vid = (face_vid - self.mean)/self.std
        face_vid = resampler(face_vid.unsqueeze(0))[0]
        face_vid = torch.transpose(face_vid, 0, 1)
        
        flipped_vid = flipped_vid.transpose((3,0,1,2))
        flipped_vid = torch.from_numpy(flipped_vid)
        flipped_vid = (flipped_vid - self.mean)/self.std
        flipped_vid = resampler(flipped_vid.unsqueeze(0))[0]
        flipped_vid = torch.transpose(flipped_vid, 0, 1)
        
        resampler = torch.nn.Upsample(size=(self.frame_length,), mode='nearest')
        # label = label[start_point:start_point+duration]
        
        if self.task == 'exp':
            label = torch.FloatTensor(label)
            label = resampler(label.view(1, 1, -1))
            label = label.long()
        elif self.task == 'va':
            label = np.asarray(label, dtype=np.float32)
            label = torch.FloatTensor(label)
            label = resampler(label.view(1, 2, -1))
        elif self.task == 'au':
            label = np.asarray(label, dtype=np.float32)
            label = torch.FloatTensor(label)
            label = resampler(label.view(1, 12, -1))
            label = label.long()

        return face_vid, flipped_vid, label

def build_seq_dataset(args, mode="train", task="exp"):
    if mode == "train":
        print("Dataloader building... [Train set]")
        dataset = image_loader(args, data_dir='/workspace/ABAW8', label_dir=args.train_csv_path, \
            frame_length=args.clip, mode="train", task=task)
    elif mode =="valid":
        print("Dataloader building... [Val set]")
        dataset = image_loader(args, data_dir='/workspace/ABAW8', label_dir=args.valid_csv_path, \
            frame_length=args.clip, mode="valid", task=task)
            
    return dataset


def build_seq_dataset_erasing(args, mode="train", task="exp"):
    if mode == "train":
        print("Dataloader building... [Train set]")
        dataset = image_loader_erasing(args, data_dir='/workspace/ABAW8', label_dir=args.train_csv_path, \
            frame_length=args.clip, mode="train", task=task)
    elif mode =="valid":
        print("Dataloader building... [Val set]")
        dataset = image_loader_erasing(args, data_dir='/workspace/ABAW8', label_dir=args.valid_csv_path, \
            frame_length=args.clip, mode="valid", task=task)
            
    return dataset

class DDPImbalancedDatasetSampler(ImbalancedDatasetSampler):
    def set_epoch(self, epoch):
        self.epoch = epoch
# if __name__ == '__main__':
#     # train_transform = build_transform(True)
    
#     transform = transforms.Compose([
#         transforms.Resize([112, 112]),
#         transforms.RandomRotation(45),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
#         transforms.ToTensor()])

#     # ABAW5Seqdata(config["train_img_path"], transform)
#     dataset = video_loader("../../cropped_data/", "../../EXPR_Classification_Challenge/Train_Set", transform=transform)
#     data = dataset.__getitem__(0)



