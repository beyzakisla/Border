# Image Processing Project

## Description

This project provides a Python script for processing images in bulk. It handles Unicode filenames and performs various image processing tasks using OpenCV and NumPy. The script reads images from an input directory, applies color filtering, morphological operations, edge detection, and contour extraction, and then saves the processed images to an output directory.

## Features

- **Unicode Filename Support**: Reads and writes image files with Unicode characters in their filenames.
- **Color Filtering**: Converts images to HSV color space and applies color range filtering.
- **Morphological Operations**: Cleans up the mask using morphological closing.
- **Edge Detection**: Detects edges using the Canny algorithm.
- **Contour Extraction**: Finds and draws contours on the processed images.
- **Batch Processing**: Processes all images in nested input folders and saves results in corresponding output folders.

## Installation

1. **Clone the Repository**

   ` git clone https://github.com/yourusername/image-processing-project.git `

2. **Navigate to the Project Directory**

   ` cd image-processing-project `

3. **Install Dependencies**

   Ensure you have Python installed. Install the required Python packages using pip:

   ` pip install opencv-python numpy `

## Usage

1. **Prepare Your Directories**

   - Place your input images in the `input` folder. You can organize images into subfolders as needed.
   - Ensure there is an `output` folder at the root of the project to store processed images. The script will create subfolders inside `output` corresponding to those in `input`.

2. **Run the Script**

   Execute the Python script using the following command:

   ` python your_script_name.py `

   Replace `your_script_name.py` with the actual name of your Python script.

3. **View Processed Images**

   After running the script, the processed images will be available in the `output` folder, organized in the same structure as the `input` folder.

## Code Overview

Below is a brief overview of the main functions in the script:

- **imread_unicode**

  Reads an image from a file with Unicode characters in the filename.

  ```
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
  ```

- **imwrite_unicode**

  Writes an image to a file with Unicode characters in the filename.

  ```
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
  ```

- **process_image**

  Processes a single image by applying color filtering, morphological operations, edge detection, and contour extraction.

  ```
  def process_image(input_path, output_path):
      print(f'{input_path} işleniyor...')
      image = imread_unicode(input_path)
      if image is None:
          print(f'{input_path} okunamadı.')
          return
      image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

      lower_bound = np.array([80, 30, 30])
      upper_bound = np.array([140, 255, 255])
      mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
      kernel = np.ones((5, 5), np.uint8)
      cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

      edges_canny = cv2.Canny(cleaned_mask, 100, 200)
      thresh = cv2.threshold(edges_canny, 127, 255, cv2.THRESH_BINARY)[1]
      contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      contours = contours[0] if len(contours) == 2 else contours[1]
      result = np.zeros_like(image_hsv)
      cv2.drawContours(result, contours, -1, (255,255,255), 3)

      thresh2 = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)[1]
      thresh2 = cv2.cvtColor(thresh2, cv2.COLOR_BGR2GRAY)
      contours2 = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      contours2 = contours2[0] if len(contours2) == 2 else contours2[1]
      result2 = np.zeros_like(image_hsv)
      cv2.drawContours(result2, contours2, -1, (255,0,0), cv2.FILLED)

      imwrite_unicode(output_path, result2)
  ```

- **process_images_in_folders**

  Processes all images within the input directory and saves the results to the output directory.

  ```
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
  ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Contact

For any questions or suggestions, please contact [your.email@example.com](mailto:your.email@example.com).

# Acknowledgements

- [OpenCV](https://opencv.org/) for the powerful computer vision library.
- [NumPy](https://numpy.org/) for numerical operations.