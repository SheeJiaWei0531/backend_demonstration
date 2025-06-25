import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv 
import numpy as np
import os, math, traceback, base64, hashlib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
project_dir = Path(__file__).absolute().parent.parent.absolute().as_posix()
video_folder = os.path.join(project_dir, "video")

def save_base64_to_mp4(b64str):
    
    if "," in b64str:
        b64str = b64str.split(",", 1)[1]
    
    video_bytes = base64.b64decode(b64str)
    video_id = hashlib.md5(video_bytes).hexdigest()
    video_name = f"{video_id}.mp4"
    save_path = os.path.join(video_folder, video_name)
    
    with open(save_path, "wb") as f:
        f.write(video_bytes)
         
    return save_path

        
class FaceLandmarkers:
    
    def __init__(self):
        model_path= os.path.join(project_dir, "models", "face", "face_landmarker.task")
        base_options = python.BaseOptions(model_asset_path= model_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes= False,
                                            output_facial_transformation_matrixes= False,
                                            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        outline_mediapipe_index = [151, 10, 67, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 297]
        mouth_mediapipe_index = [61, 78, 39, 37, 0, 267, 269, 291, 308, 321, 314, 17, 84, 91, 81, 13, 311, 402, 14, 178]
        left_eye_mediapipe_index = [362, 384, 385, 386, 387, 263, 373, 374, 380, 381, 473]
        right_eye_mediapipe_index = [33, 161, 160, 158, 157, 133, 153, 145, 144, 163, 468]
        left_eye_brown_mediapipe_index = [296, 334, 293, 300, 383, 276, 283, 282, 295]
        right_eye_brown_mediapipe_index = [66, 105, 63, 70, 156, 46, 53, 52, 65]
        nose_mediapipe_index = [6, 195, 4, 1, 220, 48, 98, 97, 94, 326, 327, 278, 440]
        complete_index = []
        complete_index.extend(outline_mediapipe_index)
        complete_index.extend(mouth_mediapipe_index)
        complete_index.extend(left_eye_mediapipe_index)
        complete_index.extend(right_eye_mediapipe_index)
        complete_index.extend(left_eye_brown_mediapipe_index)
        complete_index.extend(right_eye_brown_mediapipe_index)
        complete_index.extend(nose_mediapipe_index)
        if len(complete_index) != 106:
            raise ValueError("Not 106 landmarks")
        self.own_index = complete_index
        
    def get_face_box(self, img_bgr):
        img_height, img_width = img_bgr.shape[:2]
        try:
            img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data= img_rgb)
            results = self.detector.detect(mp_image)
            try:
                start_x = math.inf
                end_x = 0
                start_y = math.inf
                end_y = 0
                 
                if results.face_landmarks[0] is not None:
                    for landmark in results.face_landmarks[0]:
                        x, y = int(landmark.x * img_width), int(landmark.y * img_height)
                        # cv.circle(img_rgb, (x, y), 5, (0, 255, 0), -1)
                        start_x, end_x = min(start_x, x), max(end_x, x)
                        start_y, end_y = min(start_y, y), max(end_y, y)  
                              
                if start_x < 0:
                    width = end_x
                    start_x = 0
                else:
                    width = end_x - start_x
                    
                if start_y < 0:
                    height = end_y
                    start_y = 0
                else:
                    height = end_y - start_y
                    
                return start_x, start_y, width, height
                
            except Exception as e:
                return None
                           
        except Exception as e:
            return None
        

    def get_face(self, img_bgr, img_channel = 'rgb'):
        img_bgr = img_bgr.copy()
        try:
            start_x, start_y, width, height = self.get_face_box(img_bgr= img_bgr)
            end_x = start_x + width
            end_y = start_y + height
            
            face_bgr = img_bgr[start_y: end_y, start_x: end_x]
            
            if img_channel == 'rgb':
                face_rgb = cv.cvtColor(face_bgr, cv.COLOR_BGR2RGB)
                return face_rgb
            else:
                return face_bgr
                      
        except Exception as e:
            return None
        
    def get_result(self, img_bgr):
        try:
            img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data= img_rgb)
            results = self.detector.detect(mp_image)
            if len(results.face_landmarks) != 1:
                raise ValueError("Number of face is not 1")
            return results.face_landmarks[0]
             
        except Exception as e: 
            return None
        
    def get_area(self, img_bgr, start_index: int = 54, end_index : int = 397):
        """_summary_

        Args:
            img_bgr (numpy array):img in bgr channel
            start_index (int): index of mediapipe 478, this index refering to the start_x, start_y
            end_index (int): index of mediapipe 478, this index refering to the end_x, end_y

        Returns:
            img: img of cutting out section based on the start_index and end_index provided. 
        """
        img_bgr = img_bgr.copy()
        height, width = img_bgr.shape[0], img_bgr.shape[1]
        try:
            results = self.get_result(img_bgr= img_bgr)
            start_x, start_y = int(results[start_index].x * width), int(results[start_index].y * height)
            end_x, end_y = int(results[end_index].x * width), int(results[end_index].y * height)
            return img_bgr[start_y: end_y, start_x: end_x]
        
        except Exception:
            return None
        
    def get_own_pfld_106(self, img_bgr, normalize = False):
        height, width = img_bgr.shape[:2]
        try:
            results = self.get_result(img_bgr= img_bgr)
            landmarks_list = dict()
            for own_index, mediapipe_index in enumerate(self.own_index):
                landmarks = results[mediapipe_index]
                landmarks_list[own_index] = dict()
                if normalize:
                    landmarks_list[own_index]['x'] = landmarks.x
                    landmarks_list[own_index]['y'] = landmarks.y
                else:
                    landmarks_list[own_index]['x'] = int(landmarks.x * width)
                    landmarks_list[own_index]['y'] = int(landmarks.y * height)
                    
            return landmarks_list
            
        except Exception as e:
            return None
        
    def plot_own_index(self, img_bgr, selected_indices = 'all', img_channel = 'rgb', style = 'index', point_thickness = -1,  word_size = 0.5, color = (0, 0, 255)):
        img_bgr = img_bgr.copy()
        try:
            results = self.get_own_pfld_106(img_bgr = img_bgr, normalize = False)
            if selected_indices == 'all':
                selected_index = [i for i in range(len(results))]
            else:
                selected_index = selected_indices
            for index in selected_index:
                x, y = results[index].get('x'), results[index].get('y')
                if style == 'index':
                    cv.putText(img_bgr, str(index), (x, y), cv.FONT_HERSHEY_SIMPLEX, 
                        fontScale = word_size, color= (0, 0, 255), thickness= 1)  
                else:
                    cv.circle(img_bgr, (x, y), radius = 1, color = color, thickness= point_thickness)
                    
            if img_channel == 'rgb':
                img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
                return img_rgb
            else:
                return img_bgr
            
        except Exception as e:
            return None
        
    def distance(self, p1, p2 = (0,0)):
            """ 
            Return a value.
            Is a distance calculator.
            p1 should be a tuple of (x, y)
            """    
            
            return math.sqrt(((p1[0]- p2[0])**2)+ ((p1[1]- p2[1])**2))
    
    def get_EAR_left(self, img_bgr):
        
        landmarks = self.get_own_pfld_106(img_bgr, normalize = True)

        p1 = (landmarks[53]['x'], landmarks[53]['y'])
        p2 = (landmarks[54]['x'], landmarks[54]['y'])
        p3 = (landmarks[57]['x'], landmarks[57]['y'])
        p4 = (landmarks[58]['x'], landmarks[58]['y'])
        p5 = (landmarks[59]['x'], landmarks[59]['y'])
        p6 = (landmarks[62]['x'], landmarks[62]['y'])
        part1 = self.distance(p1=p2, p2=p6)
        part2 = self.distance(p1=p3, p2=p5)
        part3 = self.distance(p1=p1, p2=p4)
        ear = (part1 + part2) / (2.0 * part3)
        
        return round(ear, 3) 
    
    def get_EAR_right(self, img_bgr):
            
        landmarks = self.get_own_pfld_106(img_bgr, normalize = True)
        p1 = (landmarks[64]['x'], landmarks[64]['y'])
        p2 = (landmarks[65]['x'], landmarks[65]['y'])
        p3 = (landmarks[67]['x'], landmarks[67]['y'])
        p4 = (landmarks[69]['x'], landmarks[69]['y'])
        p5 = (landmarks[70]['x'], landmarks[70]['y'])
        p6 = (landmarks[73]['x'], landmarks[73]['y'])
        part1 = self.distance(p1=p2, p2=p6)
        part2 = self.distance(p1=p3, p2=p5)
        part3 = self.distance(p1=p1, p2=p4)
        ear = (part1 + part2) / (2.0 * part3)
        
        return round(ear, 3)
    
    def get_MAR(self, img_bgr):
        
        landmarks = self.get_own_pfld_106(img_bgr, normalize = True)
        p1 = (landmarks[33]['x'], landmarks[33]['y'])
        p2 = (landmarks[47]['x'], landmarks[47]['y'])
        p3 = (landmarks[49]['x'], landmarks[49]['y'])
        p4 = (landmarks[40]['x'], landmarks[40]['y'])
        p5 = (landmarks[50]['x'], landmarks[50]['y'])
        p6 = (landmarks[52]['x'], landmarks[52]['y'])
    
        part1 = self.distance(p1=p2, p2=p6)
        part2 = self.distance(p1=p3, p2=p5)
        part3 = self.distance(p1=p1, p2=p4)

        # Calculate EAR, Eya aspect ratio.
        mar = (part1 + part2) / (2.0 * part3)
        
        return round(mar, 3)
    
    def get_YAW(self, img_bgr):
        
        landmarks = self.get_own_pfld_106(img_bgr, normalize = True)
        yaw = abs(landmarks[95]['x'] - landmarks[25]['x']) / abs(landmarks[25]['x'] - landmarks[9]['x'])
        
        return round(yaw, 3)



        