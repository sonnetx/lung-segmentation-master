# %%
import torch
import torchvision
import os
import glob
import time 
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

from src.data import LungDataset, blend, Pad, Crop, Resize
from src.models import UNet, PretrainedUNet
from src.metrics import jaccard, dice

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

# %% [markdown]
# ## Inference
# 

# %%
unet = PretrainedUNet(
    in_channels=1,
    out_channels=2, 
    batch_norm=True, 
    upscale_mode="bilinear"
)

# %%
model_name = "unet-6v.pt"
models_folder = "models/"
unet.load_state_dict(torch.load(models_folder + model_name, map_location=torch.device("cpu")))
unet.to(device)
unet.eval()

# %%
directory = "/home/groups/roxanad/sonnet/lung-segmentation-master/inputs"
# directory = "C:\\Users\\sonne\\Documents\\GitHub\\lung-segmentation-master\\inputs"
# output_directory = "C:\\Users\\sonne\\Documents\\GitHub\\lung-segmentation-master\\masks"
output_directory = "/home/groups/roxanad/sonnet/lung-segmentation-master/masks"

# Iterate through all files in the directory
for filename in os.listdir(directory):
    if filename.lower().endswith(".jpg"):
        origin_filename = os.path.join(directory, filename)
        try:
            origin = Image.open(origin_filename).convert("P")
        except:
            continue
        origin = torchvision.transforms.functional.resize(origin, (512, 512))
        origin = torchvision.transforms.functional.to_tensor(origin) - 0.5

        with torch.no_grad():
            origin = torch.stack([origin])
            origin = origin.to(device)
            out = unet(origin)
            softmax = torch.nn.functional.log_softmax(out, dim=1)
            out = torch.argmax(softmax, dim=1)
            
            origin = origin[0].to("cpu")
            out = out[0].to("cpu")

            # Convert the binary mask to a PIL image
            binary_mask = (out.numpy() * 255).astype(np.uint8)
            binary_mask_image = Image.fromarray(binary_mask)

            # Save the binary mask image
            binary_mask_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_binary_mask.png")
            binary_mask_image.save(binary_mask_path)

# %%

def process_image(file_path):
    # Load the segmentation map
    segmentation_map = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Threshold to create a binary image (assuming lungs are represented by a specific value, e.g., > 0)
    _, binary_map = cv2.threshold(segmentation_map, 1, 255, cv2.THRESH_BINARY)

    # Find all contours in the binary map
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    height, width = binary_map.shape
    mid_point = width // 2
    
    left_lung_contours = []
    right_lung_contours = []

    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            if cx < mid_point:
                left_lung_contours.append(contour)
            else:
                right_lung_contours.append(contour)

    results = []

    for lung_contours, lung_name in zip([left_lung_contours, right_lung_contours], ['left_lung', 'right_lung']):
        if lung_contours:
            # Find the largest contour
            largest_contour = max(lung_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            results.append((lung_name, w, h))

    return results


def process_directory(directory_path, output_csv_path):
    results = []
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_results = process_image(file_path)
            for lung_name, width, height in image_results:
                results.append({
                    'file_name': file_name,
                    'lung': lung_name,
                    'width': width,
                    'height': height
                })

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)

output_csv = 'lung_dimensions.csv'

# Process the directory and save results to the CSV file
process_directory(output_directory, output_csv)
