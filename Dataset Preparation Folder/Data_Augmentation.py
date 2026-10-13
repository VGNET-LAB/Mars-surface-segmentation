import os
import cv2
import numpy as np
import albumentations as A
from albumentations.augmentations import transforms
from albumentations.core.composition import OneOf
from tqdm import tqdm

# Paths
image_dir = r"D:\u net\image_dir"
mask_dir = r"D:\u net\mask_dir"

# Output directories
aug_image_dir = r"D:\u net\augmented_images"
aug_mask_dir = r"D:\u net\augmented_masks"
os.makedirs(aug_image_dir, exist_ok=True)
os.makedirs(aug_mask_dir, exist_ok=True)

# Define augmentation pipeline
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomCrop(height=256, width=256, p=0.3),  # Optional: match your model input
    A.ElasticTransform(p=0.2)
])

# Number of augmentations per image
num_augmentations = 5

# Process each image-mask pair
for filename in tqdm(os.listdir(image_dir)):
    if filename.endswith(".png"):
        img_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename.replace(".png", "_mask.png"))

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Skipped: {filename} (missing image or mask)")
            continue

        for i in range(num_augmentations):
            augmented = augment(image=image, mask=mask)
            aug_img = augmented['image']
            aug_mask = augmented['mask']

            img_name = filename.replace(".png", f"_aug{i}.png")
            mask_name = filename.replace(".png", f"_aug{i}_mask.png")

            cv2.imwrite(os.path.join(aug_image_dir, img_name), aug_img)
            cv2.imwrite(os.path.join(aug_mask_dir, mask_name), aug_mask)

print("Augmentation complete!")
