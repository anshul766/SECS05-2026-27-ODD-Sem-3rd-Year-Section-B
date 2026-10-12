# Name: Anshul HUdda
# BTech Cse 3rd year
# sec:"B"
# Cu24250002
# ROll no. = 06

# Experiment No. 5

# Experiment Name
# Feature Extraction and Image Analysis using SIFT and HOG Descriptors

# Aim
# To implement Scale-Invariant Feature Transform (SIFT) and Histogram of Oriented Gradients (HOG) techniques for feature extraction and analyze their effectiveness in image representation, object recognition, and computer vision applications.

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import exposure

print("Libraries imported successfully!")

from google.colab import files

uploaded = files.upload()

image_name = list(uploaded.keys())[0]

image = cv2.imread(image_name)

print("Image loaded:", image_name)
print("Image shape:", image.shape)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 6))
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(8, 6))
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()

sift = cv2.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(gray, None)

print("Number of keypoints detected:", len(keypoints))
print("Descriptor shape:", descriptors.shape if descriptors is not None else None)

sift_image = cv2.drawKeypoints(
    image_rgb,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(10, 7))
plt.imshow(sift_image)
plt.title("SIFT Keypoints")
plt.axis("off")
plt.show()

print("SIFT Feature Extraction Results")
print("--------------------------------")
print("Total Keypoints:", len(keypoints))
print("Descriptor Dimensions:", descriptors.shape[1])
print("Total Descriptor Vectors:", descriptors.shape[0])

hog_image = cv2.resize(gray, (256, 256))

plt.figure(figsize=(6, 6))
plt.imshow(hog_image, cmap="gray")
plt.title("Resized Image for HOG")
plt.axis("off")
plt.show()

features, hog_visualization = hog(
    hog_image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True,
    block_norm='L2-Hys'
)

print("HOG feature extraction completed!")
print("Number of HOG features:", len(features))

hog_rescaled = exposure.rescale_intensity(
    hog_visualization,
    in_range=(0, 10)
)

plt.figure(figsize=(8, 6))
plt.imshow(hog_rescaled, cmap="gray")
plt.title("HOG Visualization")
plt.axis("off")
plt.show()

print("SIFT vs HOG")
print("=" * 50)

print("SIFT Keypoints:", len(keypoints))
print("SIFT Descriptor Size:", descriptors.shape if descriptors is not None else None)
print("HOG Feature Vector Size:", len(features))

print("Upload a second similar image")

uploaded2 = files.upload()

image_name2 = list(uploaded2.keys())[0]

image2 = cv2.imread(image_name2)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

print("Second image loaded:", image_name2)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print("Image 1 keypoints:", len(kp1))
print("Image 2 keypoints:", len(kp2))

bf = cv2.BFMatcher()

matches = bf.knnMatch(des1, des2, k=2)

good_matches = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print("Total matches:", len(matches))
print("Good matches:", len(good_matches))

matched_image = cv2.drawMatches(
    image_rgb,
    kp1,
    cv2.cvtColor(image2, cv2.COLOR_BGR2RGB),
    kp2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(16, 8))
plt.imshow(matched_image)
plt.title("SIFT Feature Matching")
plt.axis("off")
plt.show()

print("FINAL RESULT")
print("=" * 50)

print("SIFT Keypoints in Image 1:", len(kp1))
print("SIFT Keypoints in Image 2:", len(kp2))
print("Good SIFT Matches:", len(good_matches))
print("HOG Feature Vector Length:", len(features))

print("\nExperiment completed successfully.")

# EXPERIMENT 5
# Feature Extraction and Image Analysis using SIFT and HOG Descriptors
# Q1. What is feature extraction, and why is it important in computer vision?
# Answer: Feature extraction is the process of identifying important information such as edges, corners, shapes, and textures from an image. It converts raw image data into useful features that can be easily analyzed. It is important for tasks such as image classification, object detection, recognition, and image matching.

# Q2. Explain the working principle of Scale-Invariant Feature Transform (SIFT).
# Answer: SIFT detects distinctive keypoints in an image and creates numerical descriptors for them. It analyzes the image at different scales and assigns an orientation to each keypoint. This makes SIFT features robust to changes in image size and rotation, which is useful for image matching and object recognition.

# Q3. What are keypoints and feature descriptors in image analysis?
# Answer: Keypoints are distinctive points in an image, such as corners, edges, or textured regions. Feature descriptors are numerical representations of the image region surrounding each keypoint. They are used to compare corresponding features between two or more images.

# Q4. Explain the concept of Histogram of Oriented Gradients (HOG) and its significance.
# Answer: HOG stands for Histogram of Oriented Gradients. It represents an image by calculating the direction and magnitude of gradients in small regions called cells. HOG is useful for describing the shape and edge structure of objects, especially in object detection applications.

# Q5. Compare SIFT and HOG based on robustness, computational complexity, and practical applications.
# Answer: SIFT is highly robust to changes in scale and rotation, but it is comparatively more computationally expensive. HOG is mainly used to describe the shape and edges of objects and is generally simpler to compute. SIFT is commonly used for image matching, while HOG is widely used for object detection.

# Q6. Why is SIFT considered invariant to scale and rotation?
# Answer: SIFT detects keypoints at different image scales, so the same feature can be detected even when an object is resized. It also assigns a dominant orientation to each keypoint. Therefore, SIFT can recognize the same feature even when the image is rotated.

# Q7. Mention three real-world applications where HOG descriptors are commonly used.
# Answer: Three common real-world applications of HOG descriptors are:

# Pedestrian detection in surveillance systems.
# Object detection in computer vision systems.
# Human shape and activity analysis in images and videos.
# Q8. Why is feature extraction performed before image classification or object detection?
# Answer: Feature extraction converts raw image data into meaningful and useful information. It removes unnecessary details and highlights important patterns such as edges, shapes, and textures. This helps classification and object detection algorithms process images more efficiently.

# Q9. What are the advantages and limitations of handcrafted feature descriptors compared to deep learning-based feature extraction?
# Answer: Handcrafted descriptors such as SIFT and HOG are easier to understand and can work well with smaller datasets. However, they require manually designed methods and may not perform well on complex images. Deep learning methods can automatically learn more complex features from large datasets.

# Q10. How do feature extraction techniques contribute to image matching, face recognition, and object detection systems?
# Answer: Feature extraction provides meaningful representations of images that can be compared and analyzed. SIFT descriptors can be used to find matching features between images. Shape, edge, and gradient features can also help in face recognition and object detection by identifying important patterns in an image.

