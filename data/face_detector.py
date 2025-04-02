import cv2
import numpy as np
import onnxruntime
import data.face_utils as face_utils


class Face_detector():
    MEAN = np.array([127, 127, 127])

    def __init__(self):
        # Predict model
        # self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.providers = ['CPUExecutionProvider']
        self.LD_session = onnxruntime.InferenceSession('./data/fd_model/TDDFA.onnx', providers=self.providers)
        self.FACE_sess = onnxruntime.InferenceSession('./data/fd_model/facedetector.onnx', providers=self.providers)
        self.face_img = np.zeros((256,256,3), dtype=np.uint8)
        # Reset buffers
        self.reset()

    def reset(self):
        self.pre_wh = [0, 0]
        self.pre_loc = [0, 0]
        self.TDDFA_que = []
        self.box = [0,0,0,0]
        self.face_img = np.zeros((256,256,3), dtype=np.uint8)
    
    def detection(self, frame, mode="closest", margin=0, direction="front"):
        temp = frame.copy()
        height, width, _ = frame.shape
        # preprocess for detecting face box
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # CV BGR -> RGB(모델에 맞게)
        image = cv2.resize(image, (320, 240))
        image = (image - self.MEAN) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        
        # run face detector
        confidences, boxes = self.FACE_sess.run(None, {"input": image})
        # post-processing
        boxes, labels, probs = face_utils.predict(frame.shape[1], frame.shape[0], confidences, boxes, 0.5)
        det_inf = zip(boxes, probs)
        # Select face position (Choose one in nearest, right, and left face box)
        
        if direction=="front":
            det_inf = sorted(det_inf, key=lambda x: (abs(x[0][2] - x[0][0])), reverse=True)
        elif direction=="right":
            det_inf = sorted(det_inf, key=lambda x: (abs(x[0][0])), reverse=True)
        elif direction=="left":
            det_inf = sorted(det_inf, key=lambda x: (abs(x[0][0])), reverse=False)
            
        # if sum(self.pre_loc) > 0:
        #     # x축상 이전 프레임과 가까운 곳을 재검출
        #     det_inf = sorted(det_inf, key=lambda x: (abs((x[0][2]+x[0][0])/2 - self.pre_loc[0]), x[1]))
            
        if len(det_inf) == 0:
            if self.face_img is not None:
                return self.face_img
            else:
                return -1
            
        self.box = det_inf[0][0]

        # add box margin for accurate face landmark inference
        sx = self.box[0] if self.box[0] > 0 else 0
        sy = self.box[1] if self.box[1] > 0 else 0
        ex = self.box[2] if self.box[2] < width else width - 1
        ey = self.box[3] if self.box[3] < height else height - 1
        
        face_img = cv2.cvtColor(frame[sy:ey, sx:ex].copy(), cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, (320, 240))
        TDDFA_vers = face_utils.TDDFA_detector(self.LD_session, face_img, sx, sy, ex, ey)
            
        self.TDDFA_que.append(TDDFA_vers.copy())
        if len(self.TDDFA_que) >= 3:
            self.TDDFA_que = self.TDDFA_que[-3:]
            TDDFA_vers = np.mean(self.TDDFA_que, axis=0)
            
        # ROI 안정화 코드(이전 위치와 현재 위치의 차이가 없으면 이전 위치 및 바운딩 박스 크기를 사용함)
        # 기준: 1) 가로해상도 2%, 세로해상도 2% 이내 위치 차이는 반영하지 않음(경험적 세팅)
        #      2) 얼굴 움직임으로 인해 양옆 얼굴 일부가 포함되지 않아 바운딩박스의 가로너비를 120%로 넓힘(경험적 세팅)
        mean_vers = np.mean(TDDFA_vers, axis=0)
        loc = [round(mean_vers[0]), round(mean_vers[1])]
        # loc = [round(TDDFA_vers[30][0]), round(TDDFA_vers[30][1])]
        
        if abs(self.pre_wh[0] - (ex-sx)) > (width * 0.05) and \
            abs(self.pre_wh[1] - (ey-sy)) > (height * 0.05): # 기준 1)
            self.pre_wh = [int(abs(ex - sx)), int(abs(ey - sy))]
            
        if sum(self.pre_wh) == 0:
            self.pre_wh = [int(abs(ex - sx)), int(abs(ey - sy))]
        
        bbox_w = int((self.pre_wh[0] / 2)) # 기준 2)
        bbox_h = int(self.pre_wh[1] / 2)
        sx = loc[0] - bbox_w if loc[0] - bbox_w > 0 else 0
        ex = loc[0] + bbox_w if loc[0] + bbox_w < width else width - 1
        sy = loc[1] - bbox_h if loc[1] - bbox_h > 0 else 0
        ey = loc[1] + bbox_h if loc[1] + bbox_h < height else height - 1
        
        self.pre_loc = loc
        self.face_img = temp[sy:ey, sx:ex].copy()
        # cv2.circle(temp, (loc[0], loc[1]), 3, (0, 0, 255), -1, cv2.LINE_AA)
        # cv2.rectangle(temp, (sx, sy), (ex, ey), (0,0,255), 2)
        # cv2.imshow("frame", temp)
        
        # if not cropped_face:
        #     cropped_face = np.zeros((256,256,3), dtype=np.uint8)
            
        return self.face_img
        