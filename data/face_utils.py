# coding: utf-8

import cv2
import os
import numpy as np
import pickle

def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr

def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]

def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))

def conv2d_np(image, kernel):
    kernel = np.flipud(np.fliplr(kernel)) #XCorrel
    sub_matrices = np.lib.stride_tricks.as_strided(image,
                                                   shape = tuple(np.subtract(image.shape, kernel.shape))+kernel.shape, 
                                                   strides = image.strides * 2)

    return np.einsum('ij,klij->kl', kernel, sub_matrices)

def _parse_param(param):
    """matrix pose form
    param: shape=(trans_dim+shape_dim+exp_dim,), i.e., 62 = 12 + 40 + 10
    """

    # pre-defined templates for parameter
    trans_dim, shape_dim, exp_dim = 12, 40, 10

    R_ = param[:trans_dim].reshape(3, -1)
    R = R_[:, :3]
    offset = R_[:, -1].reshape(3, 1)
    alpha_shp = param[trans_dim:trans_dim + shape_dim].reshape(-1, 1)
    alpha_exp = param[trans_dim + shape_dim:].reshape(-1, 1)

    return R, offset, alpha_shp, alpha_exp

# Load 3DDFA(facial detection model) parameter
if os.path.exists('/workspace/ABAW8/ABAW_8th_CODE/data/fd_model/bfm_slim.pkl'):
    bfm = _load('/workspace/ABAW8/ABAW_8th_CODE/data/fd_model/bfm_slim.pkl')
    r = _load('/workspace/ABAW8/ABAW_8th_CODE/data/fd_model/param_mean_std_62d_120x120.pkl')
else:
    bfm = _load('/workspace/ABAW8/ABAW_8th_CODE/data/fd_model/bfm_slim.pkl')
    r = _load('/workspace/ABAW8/ABAW_8th_CODE/data/fd_model/param_mean_std_62d_120x120.pkl')

u = bfm.get('u').astype(np.float32)  # fix bug
w_shp = bfm.get('w_shp').astype(np.float32)[..., :50]
w_exp = bfm.get('w_exp').astype(np.float32)[..., :12]
tri = bfm.get('tri')
tri = _to_ctype(tri.T).astype(np.int32)
keypoints = bfm.get('keypoints').astype(np.compat.long)  # fix bug
w = np.concatenate((w_shp, w_exp), axis=1)
w_norm = np.linalg.norm(w, axis=0)
u_base = u[keypoints].reshape(-1, 1)
w_shp_base = w_shp[keypoints]
w_exp_base = w_exp[keypoints]

# params normalization config
param_mean = r.get('mean')
param_std = r.get('std')

def PFLD_detector(session, face_box, sx, sy, ex, ey):
    img = cv2.resize(face_box, dsize=(112, 112), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
    img = img / 255.0
    # PFLD detector inference
    face_lms = session.run(None, {'input': img})[0][0]
    face_lms = face_lms.reshape(-1, 2)
    vers = pfld_transform(face_lms, [sx,sy,ex,ey])    
    
    return np.array(vers)

def TDDFA_detector(session, face_box, sx, sy, ex, ey):
    face_box = cv2.resize(face_box, dsize=(120, 120), interpolation=cv2.INTER_CUBIC)
    # Inference face detection using 3DDFA(by kunyoung)
    face_box = face_box.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
    face_box = (face_box - 127.5) / 128.0
    param = session.run(None, {'input': face_box})[0]
    param = param.flatten().astype(np.float32)
    param = param * param_std + param_mean  # re-scale
    head_vers = recon_vers([param], [[sx,sy,ex,ey]], u_base, w_shp_base, w_exp_base)[0]

    return head_vers.T[:,:2]

    # # Calc head gaze vector (by kunyoung) 얼굴 방향 벡터 계산
    # head_poly = np.array([[int(round(head_vers[0, 33])),int(round(head_vers[1, 33])),int(round(head_vers[2, 33]))], \
    #         [int(round(head_vers[0, 17])),int(round(head_vers[1, 17])),int(round(head_vers[2, 17]))], \
    #         [int(round(head_vers[0, 26])),int(round(head_vers[1, 26])),int(round(head_vers[2, 26]))]])
    
    # return head_angle(head_poly), head_vers

def recon_vers(param_lst, roi_box_lst, u_base, w_shp_base, w_exp_base):
    ver_lst = []
    for param, roi_box in zip(param_lst, roi_box_lst):
        R, offset, alpha_shp, alpha_exp = _parse_param(param)
        pts3d = R @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp). \
            reshape(3, -1, order='F') + offset
        pts3d = similar_transform(pts3d, roi_box, 120)

        ver_lst.append(pts3d)

    return ver_lst

# To calculate head gaze by 3 rigid facial point
def head_angle(poly):
    n = np.cross(poly[1,:]-poly[0,:],poly[2,:]-poly[0,:])
    norm = np.linalg.norm(n)
    if norm==0:
        raise ValueError('zero norm')
    else:
        normalised = n/norm
    
        x = np.array([1,0,0])
        y = np.array([0,1,0])
        z = np.array([0,0,1])
        
        xcost =np.dot(x,normalised)
        ycost =np.dot(y,normalised)
        zcost =np.dot(z,normalised)
    
    return np.rad2deg(np.arccos(xcost)) - 90.0, np.rad2deg(np.arccos(ycost)) - 90.0, np.rad2deg(np.arccos(zcost))

def similar_transform(pts3d, roi_box, size):
    pts3d[0, :] -= 1  # for Python compatibility
    pts3d[2, :] -= 1
    pts3d[1, :] = size - pts3d[1, :]

    sx, sy, ex, ey = roi_box
    scale_x = (ex - sx) / size
    scale_y = (ey - sy) / size
    pts3d[0, :] = pts3d[0, :] * scale_x + sx
    pts3d[1, :] = pts3d[1, :] * scale_y + sy
    s = (scale_x + scale_y) / 2
    pts3d[2, :] *= s
    pts3d[2, :] -= np.min(pts3d[2, :])
    return np.array(pts3d, dtype=np.float32)

def set_res(cap, x,y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def dist_2D(p1, p2):
    squared_dist = np.sum((p1-p2)**2, axis=0)
    dist = np.sqrt(squared_dist)
    return dist
    
# def get_right_eye_roi(image, vers):
#     # to cropped right eye roi
#     eye_ROI_w = dist_2D(np.array([vers[36,0], vers[36,1]]), np.array([vers[39,0], vers[39,1]]))
#     eye_ROI_h = dist_2D(np.array([vers[40,0], vers[40,1]]), np.array([vers[38,0], vers[38,1]]))
#     margin_w = eye_ROI_w/16
#     margin_h = eye_ROI_h/8
#     eye_sy = int(vers[37,1] - 2.2*margin_h)
#     eye_ey = int(vers[37,1] + eye_ROI_h + 2.1*margin_h)
#     eye_sx = int(vers[36,0] - 2*margin_w)
#     eye_ex = int(vers[36,0] + eye_ROI_w + 2*margin_w)
#     eye_img = image[eye_sy:eye_ey, eye_sx:eye_ex].copy()
#     return eye_img, eye_sx, eye_sy, eye_img.shape[1], eye_img.shape[0]
    
def get_right_eye_roi(image, vers):
    # to cropped right eye roi
    eye_ROI_w = dist_2D(np.array([vers[36,0], vers[36,1]]), np.array([vers[39,0], vers[39,1]]))
    eye_ROI_h = dist_2D(np.array([vers[40,0], vers[40,1]]), np.array([vers[38,0], vers[38,1]]))
    margin_w = eye_ROI_w/16
    # margin_h = eye_ROI_h/8
    eye_cy = int(round((vers[37,1] + vers[41,1])/2))
    eye_sy = int(eye_cy - 40)
    eye_ey = int(eye_cy + 20)
    # eye_sy = int(vers[37,1] - 2.2*margin_h)
    # eye_ey = int(vers[37,1] + eye_ROI_h + 2.1*margin_h)
    eye_sx = int(vers[36,0] - 2*margin_w)
    eye_ex = int(vers[36,0] + eye_ROI_w + 2*margin_w)
    eye_img = image[eye_sy:eye_ey, eye_sx:eye_ex].copy()
    return eye_img, eye_sx, eye_sy, eye_img.shape[1], eye_img.shape[0]

def draw_holistic_landmarks(image, vers, pose_lms, face_box):
    scale_x = float(1920/3840)
    scale_y = float(1080/2160)
    # Draw face box(by kunyoung) 
    cv2.rectangle(image, (int(face_box[2]*scale_x), int(face_box[0]*scale_y)), (int(face_box[3]*scale_x), int(face_box[1]*scale_y)), (0,255,0), 1, cv2.LINE_AA)
    # Draw pose_landmark(by kunyoung) 
    for i in range(11,23):
        px,py,_,conf = pose_lms[i]
        cv2.circle(image, (int(1920*px), int(1080*py)), 9, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(image, (int(1920*px), int(1080*py)), 6, (255, 0, 0), -1, cv2.LINE_AA)
    # Draw face detection(by kunyoung)
    for i in range(68):
        if i == 17 or i == 26 or i == 36 or i == 45 or i == 39 or i == 42 or i == 27 or i == 29 or i == 33:
            cv2.circle(image, (int(scale_x*vers[i,0]), int(scale_y*vers[i,1])), 2, (0, 255, 0), -1, cv2.LINE_AA)

def face_ROI(pose_lms, box_ratio, pose_roi_info):
    # pose_roi_info = [self.roi_sx, self.roi_sy, self.roi_width, self.roi_heigth]
    # Face ROI define
    roi_sx = pose_roi_info[0]
    roi_sy = pose_roi_info[1]
    roi_width = pose_roi_info[2]
    roi_heigth = pose_roi_info[3]
    box_center = int(roi_width*(pose_lms[7][0]+pose_lms[8][0])/2 + roi_sx)
    box_u = pose_lms[1][1]
    box_b = pose_lms[10][1]
    box_h = int((roi_heigth*box_ratio)*(box_b - box_u))
    sy = int((roi_heigth*box_u+roi_sy) - box_h)
    ey = int((roi_heigth*box_b+roi_sy) + box_h)
    sx = int(box_center - 1.5*box_h)
    ex = int(box_center + 1.5*box_h)
    return sy+2, ey-2, sx+2, ex-2

def draw_cali_circle(window_img, center_pt, size_factor, line_size, timer, start_time):
    cv2.circle(window_img, center_pt, int(size_factor) +1, (255,0,0), 3, cv2.LINE_AA)
    cv2.circle(window_img, center_pt, int((size_factor/4)*(timer-start_time)), (0,0,255), -1, cv2.LINE_AA)
    cv2.line(window_img, (center_pt[0] - line_size, center_pt[1]), (center_pt[0] + line_size, center_pt[1]), (0,255,0), 2, cv2.LINE_AA)
    cv2.line(window_img, (center_pt[0], center_pt[1] - line_size), (center_pt[0], center_pt[1] + line_size), (0,255,0), 2, cv2.LINE_AA)

def pfld_transform(pts, roi_box):
    sx, sy, ex, ey = roi_box
    w = (ex - sx)
    h = (ey - sy)
    landmark_= np.asarray(np.zeros(pts.shape))
    for i, point in enumerate(pts):
        x = point[0] * w + sx
        y = point[1] * h + sy
        landmark_[i] = (x, y)
    return landmark_

def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]
    
def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    # _, indexes = scores.sort(descending=True)
    indexes = np.argsort(scores)
    # indexes = indexes[:candidate_size]
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        # current = indexes[0]
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        # indexes = indexes[1:]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]