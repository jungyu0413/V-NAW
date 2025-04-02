import os
import numpy as np

# Neutral, Anger, Disgust, Fear, Happiness, Sadness, Surprise, Other
# 0, 1, 2, 3, 4, 5, 6, -1
def build_anno_list(label_dir, window_size=300, stride=150):
    vid_paths = []
    for dirpath, dirnames, filenames in os.walk(label_dir):
        for dirname in dirnames:
            if dirname.split('/')[-1] not in ["batch1", "batch2"]:
                vid_paths.append(os.path.join(dirpath, dirname))
    
    vid_paths = sorted(vid_paths)
    ret = {}
    # for path in vid_paths:
    for path in vid_paths:
        imgs = os.listdir(path)
        imgs = sorted(imgs)
        idxs = [int(idx[:-4]) for idx in imgs if idx[-4:] in ['.jpg','.JPG','.png','.PNG']]
        idxs = np.array(idxs)
        diff = idxs[:-1] - idxs[1:]
        null_idx = np.where(diff>=0)[0]
        
        if len(null_idx) > 0:
            print("[data/dataset.py build_seq()]\n Data excluded due to discontinuous video sequences!")
            print(path, null_idx)
        else:
            s_pt = idxs[0]
            e_pt = idxs[-1]
            s_pts = np.arange(s_pt, e_pt-window_size, stride) # 1, 151, 301, ...
            e_pts = np.arange(s_pt+window_size, e_pt, stride) # 301, 451, 601, ...
            seq_pts = [(s_pts[i], e_pts[i]) for i in range(len(s_pts))]
            
            name = path.split('/')[-1]
            ret[name]= [seq_pts, e_pt]
            # ret[name]= seq_pts
    
    return ret

# Neutral, Anger, Disgust, Fear, Happiness, Sadness, Surprise, Other
# 0, 1, 2, 3, 4, 5, 6, -1
def build_vid_seqs(data_dir, window_size=300, stride=150):
    vid_paths = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for dirname in dirnames:
            if dirname.split('/')[-1] not in ["batch1", "batch2"]:
                vid_paths.append(os.path.join(dirpath, dirname))
    
    vid_paths = sorted(vid_paths)
    ret = {}
    # for path in vid_paths:
    for path in vid_paths:
        imgs = os.listdir(path)
        imgs = sorted(imgs)
        idxs = [int(idx[:-4]) for idx in imgs if idx[-4:] in ['.jpg','.JPG','.png','.PNG']]
        idxs = np.array(idxs)
        diff = idxs[:-1] - idxs[1:]
        null_idx = np.where(diff>=0)[0]
        
        if len(null_idx) > 0:
            print("[data/dataset.py build_seq()]\n Data excluded due to discontinuous video sequences!")
            print(path, null_idx)
        else:
            s_pt = idxs[0]
            e_pt = idxs[-1]
            s_pts = np.arange(s_pt, e_pt-window_size, stride) # 1, 151, 301, ...
            e_pts = np.arange(s_pt+window_size, e_pt, stride) # 301, 451, 601, ...
            seq_pts = [(s_pts[i], e_pts[i]) for i in range(len(s_pts))]
            
            name = path.split('/')[-1]
            ret[name]= [seq_pts, e_pt]
            # ret[name]= seq_pts
    
    return ret


def get_vid_file(data_dir, file_name):
    
    vid_name = ""
    direction = "front"
    
    if "right" in file_name:
        file_name = file_name.replace("_right","")
        direction = "right"
        
    if "left" in file_name:
        file_name = file_name.replace("_left","")
        direction = "left"
    
    if os.path.isfile(data_dir + "/batch1/" + file_name.replace(".txt", ".mp4")):
        vid_name = data_dir + "/batch1/" + file_name.replace(".txt", ".mp4")
    if os.path.isfile(data_dir + "/batch2/" + file_name.replace(".txt", ".mp4")):
        vid_name = data_dir + "/batch2/" + file_name.replace(".txt", ".mp4")
    if os.path.isfile(data_dir + "/batch1/" + file_name.replace(".txt", ".avi")):
        vid_name = data_dir + "/batch1/" + file_name.replace(".txt", ".avi")
    if os.path.isfile(data_dir + "/batch2/" + file_name.replace(".txt", ".avi")):
        vid_name = data_dir + "/batch2/" + file_name.replace(".txt", ".avi")
        
    return vid_name, direction


def get_img_file(data_dir, file_name, val, mode):
    
    vid_name = ""
    direction = "front"
    
    if val > 0.5 or mode != "train":
        if os.path.isdir(data_dir + "/cropped_aligned/" + file_name.replace(".txt", "")):
            vid_name = data_dir + "/cropped_aligned/" + file_name.replace(".txt", "")
        if os.path.isdir(data_dir + "/cropped_aligned_new_50_vids/" + file_name.replace(".txt", "")):
            vid_name = data_dir + "/cropped_aligned_new_50_vids/" + file_name.replace(".txt", "")
        if os.path.isdir(data_dir + "/cropped_aligned/" + file_name.replace(".txt", "")):
            vid_name = data_dir + "/cropped_aligned/" + file_name.replace(".txt", "")
        if os.path.isdir(data_dir + "/cropped_aligned_new_50_vids/" + file_name.replace(".txt", "")):
            vid_name = data_dir + "/cropped_aligned_new_50_vids/" + file_name.replace(".txt", "")
        
        if vid_name == "":
            print("[File loading fail 1]", "directory:", "file_name:", file_name)
            
    else:
        if os.path.isdir(data_dir + "/cropped_enhanced/batch1/" + file_name.replace(".txt", "")):
            vid_name = data_dir + "/cropped_enhanced/batch1/" + file_name.replace(".txt", "")
        if os.path.isdir(data_dir + "/cropped_enhanced/batch2/" + file_name.replace(".txt", "")):
            vid_name = data_dir + "/cropped_enhanced/batch2/"+ file_name.replace(".txt", "")
        #jungyu
        # if vid_name == "":
        #     print("[File loading fail 2]", "directory:", "file_name:", file_name)
    # else:
    #     if os.path.isdir(data_dir + "/cropped/batch1/" + file_name.replace(".txt", "")):
    #         vid_name = data_dir + "/cropped/batch1/" + file_name.replace(".txt", "")
    #     if os.path.isdir(data_dir + "/cropped/batch2/" + file_name.replace(".txt", "")):
    #         vid_name = data_dir + "/cropped/batch2/"+ file_name.replace(".txt", "")

    
    return vid_name, direction


def build_dataset_with_label(label_dir, vid_seqs):
    txts = os.listdir(label_dir)
    txts = [txt for txt in txts if txt[-4:] == '.txt']
    
    ret = {}
    
    for txt in txts:
        name = txt[:-4]
        
        with open(os.path.join(label_dir, txt), 'r') as file:
            lines = file.readlines()
        # 각 줄 끝의 개행 문자를 제거합니다.
        lines = [int(line.strip()) if int(line.strip()) != -1 else 7 for line in lines[1:]]
        # lines = [int(line.strip()) + 1 for line in lines[1:]]
        seqs, end = vid_seqs[name]

        if end > len(lines):
            print(name, vid_seqs[name][1], len(lines))
            
        if len(lines) < end:
            while len(lines) < end:
                lines.append(lines[-1])

        labels = []
        for seq in seqs:
            labels.append(lines[seq[0]:seq[1]])
            
        ret[name] = {"frames": seqs, "labels": labels}
        
    return ret
    