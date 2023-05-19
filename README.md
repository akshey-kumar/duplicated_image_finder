# Image duplication detector
This repository provides functions that help you find duplicate images in a database. It's designed specifically for scientific plagiarism detection, allowing you to identify instances where the same image has been reused. By using these functions, you can compare images effectively and quickly detect potential duplicates. The full description of the original task can be found in task.pdf.

To run this notebook, download and unzip the data and add it to the repository:
https://drive.google.com/file/d/1Gg2eIbPBE4xpyvCiHrEzMQw4XRsAiYbk/view?usp=share_link

We offer two approaches for duplicate image detection viz. template matching and a deep learning approach (in experiments). 

usage_demo.ipynb in notebooks/ contains functions that search for duplication of a query image in a database. It is able to identify modified duplicated images even in cases where the image has been rotated, the contrast has been changed, or the image has been rescaled. The input is a query image, and the output is the file name of the source image in which the duplication appears, including the coordinates of the duplicated area [X, Y, WIDTH, HEIGHT], and the confidence of duplicate identification.

The deep learning approach (in notebooks/experiments) uses pre-trained ResNet-50 to extract features from the source images and the query image. It then performs a Nearest Neighbour search from the query feature vector in database of source feature vectors. The 5 most similar images are then short-listed.

The deep learning approach works best when the source images are decomposed in their constituent sub images. image_preprocessing.ipynb achieves this and experiments with other possible pre-preocessing strategies to enhance performance further.


