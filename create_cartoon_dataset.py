#!/usr/bin/env python3
"""
create_cartoon_dataset.py

This script:
1. Reads all images from an input folder (e.g. ./dataset/real/).
2. Applies a cartoonifying pipeline (e.g., edge preservation & thresholding).
3. Saves the cartoon images to another folder (e.g. ./dataset/cartoon/).
4. Splits both the real and cartoon images into train and test sets (e.g., 80/20).
"""

import cv2
import glob
import os
from sklearn.model_selection import train_test_split

# ==============
# CONFIGURATIONS
# ==============
INPUT_FOLDER = "./dataset/real/"        # Directory containing original (real) images
OUTPUT_FOLDER = "./dataset/cartoon/"    # Directory where cartoonized images will be saved

# Folders to split the dataset
TRAIN_REAL_DIR = "./dataset/train/real/"
TRAIN_CARTOON_DIR = "./dataset/train/cartoon/"
TEST_REAL_DIR = "./dataset/test/real/"
TEST_CARTOON_DIR = "./dataset/test/cartoon/"

# Desired image width & height
IMG_SIZE = (256, 256)

# Train-Test split ratio
TEST_SIZE = 0.2
RANDOM_SEED = 42

# =========
# FUNCTIONS
# =========

def cartoonify_image(image):
    """
    Applies an improved cartoon effect to the input image using OpenCV.
    Returns the cartoonified image with enhanced saturation for vibrant colors.
    """
    # Resize to consistent dimensions
    image = cv2.resize(image, IMG_SIZE)

    # 1. Bilateral Filtering for color smoothing
    bilateral = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. Enhance edges using median blur and adaptive thresholding
    gray_blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        gray_blur,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=9,
        C=2
    )

    # 4. Apply stronger bilateral filtering to create color segmentation
    segmented = cv2.bilateralFilter(bilateral, d=9, sigmaColor=150, sigmaSpace=150)

    # 5. Combine segmented image with edges for a cartoon effect
    cartoon = cv2.bitwise_and(segmented, segmented, mask=edges)

    return cartoon

def ensure_dir_exists(dir_path):
    """
    Creates a directory if it does not exist.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def main():
    """
    Main function to:
    1. Cartoonify all images in INPUT_FOLDER.
    2. Save them to OUTPUT_FOLDER.
    3. Perform train/test split.
    4. Move images into train/test folders accordingly.
    """
    # 1. Ensure output directories exist
    ensure_dir_exists(OUTPUT_FOLDER)
    
    # 2. Read all images from INPUT_FOLDER
    image_paths = glob.glob(os.path.join(INPUT_FOLDER, "*.*"))  # e.g. jpg, png, etc.
    image_paths.sort()
    
    # 3. Cartoonify and save
    print("[INFO] Cartoonifying images...")
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Unable to read image: {img_path}")
            continue
        
        cartoon = cartoonify_image(img)
        
        # Save cartoon image in OUTPUT_FOLDER with the same filename
        base_name = os.path.basename(img_path)
        save_path = os.path.join(OUTPUT_FOLDER, base_name)
        cv2.imwrite(save_path, cartoon)
    
    print(f"[INFO] Cartoonized images saved in: {OUTPUT_FOLDER}")
    
    # 4. Split into train and test sets
    #    We'll do it by taking the pairs: real vs. cartoon
    print("[INFO] Splitting dataset into train and test sets...")
    
    # Recollect file paths, ensuring they match by name
    real_images = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.*")))
    cartoon_images = sorted(glob.glob(os.path.join(OUTPUT_FOLDER, "*.*")))
    
    # Basic check: real_images and cartoon_images should have the same length
    if len(real_images) != len(cartoon_images):
        print("[ERROR] Mismatch in number of real vs. cartoon images.")
        return
    
    # Split
    train_real, test_real, train_cartoon, test_cartoon = train_test_split(
        real_images,
        cartoon_images,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )
    
    # Create train/test directories
    ensure_dir_exists(TRAIN_REAL_DIR)
    ensure_dir_exists(TRAIN_CARTOON_DIR)
    ensure_dir_exists(TEST_REAL_DIR)
    ensure_dir_exists(TEST_CARTOON_DIR)
    
    # Move files to respective directories
    for r, c in zip(train_real, train_cartoon):
        # Move/copy or simply rename
        r_base = os.path.basename(r)
        c_base = os.path.basename(c)
        
        # for demonstration, let's copy them (or you can move with os.rename)
        cv2.imwrite(os.path.join(TRAIN_REAL_DIR, r_base), cv2.imread(r))
        cv2.imwrite(os.path.join(TRAIN_CARTOON_DIR, c_base), cv2.imread(c))
    
    for r, c in zip(test_real, test_cartoon):
        r_base = os.path.basename(r)
        c_base = os.path.basename(c)
        
        cv2.imwrite(os.path.join(TEST_REAL_DIR, r_base), cv2.imread(r))
        cv2.imwrite(os.path.join(TEST_CARTOON_DIR, c_base), cv2.imread(c))
    
    print("[INFO] Dataset split complete!")
    print(f"      Train (real/cartoon) => {len(train_real)} images each")
    print(f"      Test (real/cartoon)  => {len(test_real)} images each")
    print("[INFO] Done.")

if __name__ == "__main__":
    main()