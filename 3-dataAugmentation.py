import os
import random
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import albumentations as A
import matplotlib.pyplot as plt

# Paths
DATA_DIR = Path(os.path.join(os.path.dirname(__file__), "data"))
RGB_DIR = DATA_DIR / "rgb"
SEM_DIR = DATA_DIR / "semantics"
AUG_RGB_DIR = DATA_DIR / "aug_rgb"
AUG_SEM_DIR = DATA_DIR / "aug_semantics"
AUG_RGB_DIR.mkdir(parents=True, exist_ok=True)
AUG_SEM_DIR.mkdir(parents=True, exist_ok=True)

# Augmentation pipeline
AUGMENTATIONS = A.Compose([
    A.HorizontalFlip(p=0.5),             # Flipping horizontally with 50% probability
    A.RandomBrightnessContrast(p=0.5),   # Adjusting brightness and contrast with 50% probability
    A.Rotate(limit=15, p=0.5),           # Rotating the image within a limit of 15 degrees with 50% probability
    A.VerticalFlip(p=0.5),               # Flipping vertically with 50% probability
    A.RandomGamma(p=0.5)                 # Adjusting gamma with 50% probability
])

# List all valid pairs
image_files = sorted([f for f in RGB_DIR.glob("*.png")])
mask_files = sorted([f for f in SEM_DIR.glob("*.npy")])

assert len(image_files) == len(mask_files), "Mismatch between RGB and mask files."

# Number of augmentations per image
AUG_PER_IMAGE = 2

# Perform augmentation
for img_path, mask_path in tqdm(zip(image_files, mask_files), total=len(image_files)):
    img = cv2.imread(str(img_path))
    mask = np.load(mask_path).astype(np.uint8)

    for i in range(AUG_PER_IMAGE):
        augmented = AUGMENTATIONS(image=img, mask=mask)
        aug_img = augmented['image']
        aug_mask = augmented['mask']

        # Save augmented files
        stem = img_path.stem + f"_aug{i}"
        cv2.imwrite(str(AUG_RGB_DIR / f"{stem}.png"), aug_img)
        np.save(str(AUG_SEM_DIR / f"{stem}.npy"), aug_mask)

print("\nData augmentation completed.")

# Optional: Visualize one sample
sample_img = cv2.imread(str(image_files[0]))
sample_mask = np.load(mask_files[0])
aug_example = AUGMENTATIONS(image=sample_img, mask=sample_mask)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
axs[0].set_title("Original")
axs[0].axis("off")
axs[1].imshow(cv2.cvtColor(aug_example['image'], cv2.COLOR_BGR2RGB))
axs[1].imshow(aug_example['mask'], alpha=0.4, cmap='jet')
axs[1].set_title("Augmented")
axs[1].axis("off")
plt.tight_layout()
plt.show()
