import os
import shutil
import argparse
import yaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def generate_images(images_dir, labels_dir):
    for image in os.listdir(images_dir):
        if os.path.isfile(os.path.join(images_dir, image)):
            with open(os.path.join(labels_dir, f'{os.path.splitext(image)[0]}.txt')) as f:
                class_index = int(f.readline().split()[0])
            yield image, class_index

def copy_images(images_dir, labels_dir, output_dir, class_names):
    for image, class_index in tqdm(generate_images(images_dir, labels_dir), desc=f"Copying images"):
        class_name = class_names[class_index]
        src_path = os.path.join(images_dir, image)
        dst_dir = os.path.join(output_dir, class_name)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src_path, os.path.join(dst_dir, image))

def main(dataset, output, split):
    # Load class names from data.yaml
    with open(os.path.join(dataset, 'data.yaml')) as f:
        data = yaml.safe_load(f)
        class_names = data['names']

    # Create train and val directories
    train_dir = os.path.join(output, 'train')
    val_dir = os.path.join(output, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Get the list of train images
    train_images_dir = os.path.join(dataset, 'train', 'images')
    train_labels_dir = os.path.join(dataset, 'train', 'labels')
    train_images = [f for f in os.listdir(train_images_dir) if os.path.isfile(os.path.join(train_images_dir, f))]

    # Copy the train images to the train directory
    copy_images(train_images, train_images_dir, train_labels_dir, train_dir, class_names)

    # Get the list of val images
    val_images_dir = os.path.join(dataset, 'valid', 'images')
    val_labels_dir = os.path.join(dataset, 'valid', 'labels')
    val_images = [f for f in os.listdir(val_images_dir) if os.path.isfile(os.path.join(val_images_dir, f))]

    # Copy the val images to the val directory
    copy_images(val_images, val_images_dir, val_labels_dir, val_dir, class_names)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the source data directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to the output directory')
    parser.add_argument('--split', type=float, default=0.2,
                        help='Proportion of the data to include in the validation set')
    args = parser.parse_args()
    main(args.dataset, args.output, args.split)
