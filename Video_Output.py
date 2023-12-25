import argparse
import cv2
from model import Generator
from PIL import Image
from torch.autograd import Variable
from utils import *
import os
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='path to video input', help='path to input video')# Input video Fog/Sand
parser.add_argument('--output', type=str, default='Output_Video/Sand_out.mkv', help='path to output video') #Change name 
parser.add_argument('--model_path', type=str, default='Trained_weights/latest_model_Sand.pt', help='path to the pre-trained model')# Change to Sand/Fog
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

my_model = Generator()
my_model.cuda()
my_model.load_state_dict(torch.load(args.model_path))
my_model.eval()

cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.output, fourcc, fps, (width * 2, height))  # Double the width for side-by-side display

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scale = 32
        frame_rgb = cv2.resize(frame_rgb, (width // scale * scale, height // scale * scale))

        frame_tensor = rgb_to_tensor(frame_rgb)
        frame_tensor = frame_tensor.unsqueeze(0)
        frame_tensor = Variable(frame_tensor.cuda())

        with torch.no_grad():
            output = my_model(frame_tensor)
        output_rgb = tensor_to_rgb(output)
        out_frame_rgb = Image.fromarray(np.uint8(output_rgb), mode='RGB')
        out_frame_rgb = out_frame_rgb.resize((width, height), resample=Image.BICUBIC)
        out_frame_bgr = cv2.cvtColor(np.array(out_frame_rgb), cv2.COLOR_RGB2BGR)
        frame= cv2.resize(frame,(600,600))
        out_frame_bgr= cv2.resize(out_frame_bgr,(600,600))
        combined_frame = np.hstack((frame,out_frame_bgr))
        out.write(combined_frame)
        cv2.imshow('Input and Dehazed Frames', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
