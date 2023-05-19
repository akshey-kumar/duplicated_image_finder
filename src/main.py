import os 
import time
import numpy as np
import pickle
import tensorflow as tf
import keras
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image
from tqdm import tqdm_notebook as tqdm

def get_imgs(root_dir, gray=False):
    extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
    filename_list = []
    img_list = []
    counter = 1
    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                file_path = os.path.join(root, filename)
                filename_list.append(file_path)
                if gray:
                    img_list.append(cv2.imread(file_path, 0))
                else: 
                    img_list.append(cv2.imread(file_path))
                counter += 1
    return np.array(filename_list), img_list

def template_matching(query_image, source_image, threshold, plot=True, method = cv2.TM_CCOEFF_NORMED):
    
    # Convert to grayscale to speedup
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY).copy()
    source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY).copy()

    # Check if query is smaller than source
    if (query_image.shape[0] > source_image.shape[0]) or (query_image.shape[1] > source_image.shape[1] ):
        return None, 0.
    
    # Perform template matching
    result = cv2.matchTemplate(source_image, query_image, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # Handling different methods, setting confidence
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        confidence = 1-min_val
        top_left = min_loc # location
    else:
        confidence = max_val
        top_left = max_loc
    
    # Set a threshold for template matching
    if confidence < threshold:
        return None, confidence  # Template not present in the source image
    else:
        return top_left, confidence


def plot_template(query_image, source_image, top_left):
    
    # Get the dimensions of the query image
    query_height, query_width = query_image.shape[:2]

    # Get the coordinates for drawing the rectangle
    bottom_right = (top_left[0] + query_width, top_left[1] + query_height)

    # Draw a rectangle around the matched template
    rect = cv2.rectangle(source_image.copy(), top_left, bottom_right, (0, 255, 0), 2)

    # Plot the source image with the rectangle
    plt.figure()
    plt.imshow(rect)
    plt.axis('off')
    plt.show()


def check_query(sources_filenames, sources_imgs, query_img):
    
    sources_with_query = []
    ### Stage 1
    for source_img, source_filename in zip(sources_imgs, sources_filenames):
        location, confidence = template_matching(query_img, source_img, threshold=0.95)
        if confidence > 0.95:
            sources_with_query.append((source_filename, location, query_img.shape, confidence))
            plot_template(query_img, source_img, location)
            return sources_with_query
            
            
    ### Stage 2
    if sources_with_query == []:
        for rotate in [cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            query_img_rt = cv2.rotate(query_img, rotate)
            for source_img, source_filename in zip(sources_imgs, sources_filenames):
                location, confidence = template_matching(query_img_rt, source_img, threshold=0.95)
                if confidence > 0.95:
                    sources_with_query.append((source_filename, location, query_img_rt.shape, confidence))
                    plot_template(query_img_rt, source_img, location)
                    return sources_with_query
                    
    return sources_with_query
    