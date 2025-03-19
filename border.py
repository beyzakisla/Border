import cv2
import numpy as np
import os

def imread_unicode(filename, flags=cv2.IMREAD_COLOR):
    try:
        with open(filename, "rb") as stream:
            bytes = bytearray(stream.read())
            nparr = np.asarray(bytes, dtype=np.uint8)
            img = cv2.imdecode(nparr, flags)
            return img
    except Exception as e:
        print(f"Hata: {e}")
        return None

def imwrite_unicode(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)
        if result:
            with open(filename, 'wb') as f:
                f.write(n.tobytes())
            return True
        else:
            return False
    except Exception as e:
        print(f"Hata: {e}")
        return False

def process_image(input_path, output_path):
    print(f'{input_path} işleniyor...')
    image = imread_unicode(input_path)
    if image is None:
        print(f'{input_path} okunamadı.')
        return
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_all = np.array([105, 0, 0])
    upper_all = np.array([125, 255, 255])
    mask = cv2.inRange(image_hsv, lower_all, upper_all)
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    grad_x = cv2.Sobel(cleaned_mask, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(cleaned_mask, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    thresh = cv2.threshold(grad, 127, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        result = np.zeros_like(image_hsv)
        cv2.drawContours(result, [largest_contour], -1, (255, 255, 255), 3)
    else:
        result = np.zeros_like(image_hsv)
        print('Hiç kontur bulunamadı.')

    imwrite_unicode(output_path, result)

def process_images_in_folders(input_folder, output_folder):
    for lake_name in os.listdir(input_folder):
        lake_input_path = os.path.join(input_folder, lake_name)
        lake_output_path = os.path.join(output_folder, lake_name)
        if not os.path.exists(lake_output_path):
            os.makedirs(lake_output_path)
        for image_name in os.listdir(lake_input_path):
            input_image_path = os.path.join(lake_input_path, image_name)
            output_image_path = os.path.join(lake_output_path, image_name)
            process_image(input_image_path, output_image_path)

input_folder = 'input'
output_folder = 'output'
process_images_in_folders(input_folder, output_folder)