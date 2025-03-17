import os
import shutil
import random

def split_dataset(input_folder, output_folder, category, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits images from a folder into train/val/test sets while maintaining category subfolders.

    Args:
        input_folder (str): Path to the folder containing original images.
        output_folder (str): Path where the split datasets will be saved.
        category (str): The category name ('abstrait' or 'figuratif').
        train_ratio (float): Percentage of images for training.
        val_ratio (float): Percentage of images for validation.
        test_ratio (float): Percentage of images for testing.
    """
    assert train_ratio + val_ratio + test_ratio == 1, "The ratios must sum up to 1."

    # Create output directories for the given category
    train_folder = os.path.join(output_folder, "train", category)
    val_folder = os.path.join(output_folder, "val", category)
    test_folder = os.path.join(output_folder, "test", category)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get list of images in the input folder
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)  # Shuffle images randomly

    # Determine the number of images for each split
    total_images = len(images)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)

    # Split images into train, val, and test sets
    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    # Copy images to their respective folders
    for img in train_images:
        shutil.copy(os.path.join(input_folder, img), os.path.join(train_folder, img))
    for img in val_images:
        shutil.copy(os.path.join(input_folder, img), os.path.join(val_folder, img))
    for img in test_images:
        shutil.copy(os.path.join(input_folder, img), os.path.join(test_folder, img))

    print(f"Dataset split completed for '{category}': {train_count} train, {val_count} val, {len(test_images)} test.")


# Split dataset into train/val/test sets
abstrait = "../../abstrait"
figuratif = "../../figuratif"
output_folder = "../../data"

split_dataset(abstrait, output_folder, "abstrait", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
split_dataset(figuratif, output_folder, "figuratif", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
