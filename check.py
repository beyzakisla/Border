import os
import cv2
import numpy as np
from tqdm import tqdm

def process_image(filepath):
    with open(filepath, 'rb') as f:
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is not None:
        total_pixels = image.shape[0] * image.shape[1]
        black_pixels = np.sum(np.all(image == [0, 0, 0], axis=-1))
        if black_pixels / total_pixels == 1:
            #print(f"Siliniyor: {filepath}, Siyah piksel oranı: {black_pixels / total_pixels:.2f}")
            os.remove(filepath)

def process_folder(lake_folder_path):
    files = [f for f in os.listdir(lake_folder_path) if f.lower().endswith(('.png', '.jpg'))]
    for filename in tqdm(files, desc=f"Processing {lake_folder_path}"):
        filepath = os.path.join(lake_folder_path, filename)
        process_image(filepath)

def main():
    main_folder = "output"
    for lake_folder in os.listdir(main_folder):
        lake_folder_path = os.path.join(main_folder, lake_folder)
        if os.path.isdir(lake_folder_path):
            process_folder(lake_folder_path)

if __name__ == "__main__":
    main()
