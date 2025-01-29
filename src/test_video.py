import torch
import numpy as np
import cv2
import albumentations
import config

from model import FaceKeypointResNet50

model = FaceKeypointResNet50(pretrained=False, requires_grad=False).to(config.DEVICE)
checkpoint = torch.load(r'D:\AI dev\ML\project\Face-Detector\outputs\model.pth',weights_only= False) #path to the model
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
if (cap.isOpened() == False):
    print('Error while trying to open webcam. Plese check again...')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

save_path = r'D:\AI dev\ML\project\Face-Detector\outputs\vid_key.mp4' #path to save the video

out = cv2.VideoWriter(f"{save_path}",
                      cv2.VideoWriter_fourcc(*'mp4v'),20,
                      (frame_width,frame_height))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        with torch.no_grad():
            image = frame
            image = cv2.resize(image, (224,224))
            orig_frame = image.copy()
            orig_h, orig_w, c = orig_frame.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image/255.0
            image = np.transpose(image, (2,0,1))
            image = torch.tensor(image, dtype=torch.float)
            image = image.unsqueeze(0).to(config.DEVICE)
            
            outputs = model(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = outputs.reshape(-1,2)
        keypoints = outputs
        for p in range(keypoints.shape[0]):
            cv2.circle(orig_frame, (int(keypoints[p,0]), int(keypoints[p,1])),
                       1,(0,0,255),-1,cv2.LINE_AA)
        
        orig_frame = cv2.resize(orig_frame, (frame_width, frame_height))
        cv2.imshow('Facial Keypoint Frame', orig_frame)
        out.write(orig_frame)
        
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break 
    else: 
        break
cap.release()
cv2.destroyAllWindows()