import os
import sys
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tqdm import tqdm

# === DEVICE SETUP ===
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print("MPS support is experimental and may yield different results.")

# === PATH SETUP ===
# Define base directories for better consistency.
base_dir = os.path.join('..', 'data')
annotations_dir = os.path.join(base_dir, 'annotations')
images_dir = os.path.join(base_dir, 'images')
xml_dir = os.path.join(annotations_dir, 'xmls')
list_file = os.path.join(annotations_dir, 'list.txt')
output_dir = './sam_masks'
os.makedirs(output_dir, exist_ok=True)

# Insert required paths
sys.path.insert(0, '/workspace/ADL-group-project/weakly_supervised/sam2')
sys.path.insert(0, os.path.abspath('sam2'))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# === LOAD MODEL ===
sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", 
                        "sam2/models/sam2.1_hiera_large.pt", 
                        device=device)
predictor = SAM2ImagePredictor(sam2_model)

# === LOAD VALID IMAGE NAMES + CENTERS ===
print("Loading valid image bounding boxes...")
valid_images = []
valid_centers = []

with open(list_file, 'r') as f:
    # Skip header lines
    for _ in range(6):
        next(f)
    image_names = [line.split()[0] for line in f]

for img_name in image_names:
    xml_path = os.path.join(xml_dir, f'{img_name}.xml')
    if os.path.exists(xml_path):
        try:
            tree = ET.parse(xml_path)
            bbox = tree.getroot().find('object/bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            valid_images.append(img_name)
            valid_centers.append([(xmin + xmax) / 2, (ymin + ymax) / 2])
        except Exception as e:
            print(f"Error parsing XML for image {img_name}: {e}")
            continue

# valid_centers is used only for point coordinates later when mode is 'center'.
valid_centers = np.array(valid_centers)
print(f"Total valid images: {len(valid_images)}")

# === CONFIGURATION ===
mode = 'triangle3'  # Options: 'center' or 'triangle3'
save_name = "single-point-sam" if mode == 'center' else "triangle3-point-sam"

# === PROCESS IMAGES ===
sample_size = len(valid_images)  # Process all valid images; adjust if needed
selected_indices = np.random.choice(len(valid_images), sample_size, replace=False)

all_masks = []   # To store per-image masks as torch tensors.
all_scores = []  # To store per-image scores as torch tensors.
processed_names = []  # To store processed image names.

print(f"Processing {sample_size} images using mode '{mode}'...")

def show_masks(image, masks, scores, point_coords=None, input_labels=None, borders=True):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.imshow(masks[0], alpha=0.5)
    if point_coords is not None:
        ax.scatter(point_coords[:, 0], point_coords[:, 1], c='lime', s=20, edgecolors='black')
    ax.set_title(f"Score: {scores[0]:.3f}")
    ax.axis('off')
    plt.show()
    plt.close()

for idx in tqdm(selected_indices):
    try:
        img_path = os.path.join(images_dir, f'{valid_images[idx]}.jpg')
        img = Image.open(img_path)
        predictor.set_image(np.array(img))

        if mode == 'center':
            point_coords = valid_centers[idx].reshape(1, -1)
            point_labels = np.array([1])
        elif mode == 'triangle3':
            xml_path = os.path.join(xml_dir, f'{valid_images[idx]}.xml')
            tree = ET.parse(xml_path)
            bbox_elem = tree.getroot().find('object/bndbox')
            xmin = int(bbox_elem.find('xmin').text)
            ymin = int(bbox_elem.find('ymin').text)
            xmax = int(bbox_elem.find('xmax').text)
            ymax = int(bbox_elem.find('ymax').text)

            center_point = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
            box_width = xmax - xmin
            box_height = ymax - ymin
            r = 0.2 * min(box_width, box_height)
            angles = [90, 210]
            angle_rad = [math.radians(a) for a in angles]
            extra_points = np.array([
                (center_point[0] + r * math.cos(a), center_point[1] - r * math.sin(a))
                for a in angle_rad
            ])
            point_coords = np.vstack([center_point, extra_points])
            point_labels = np.ones(len(point_coords), dtype=int)

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False
        )

        # Convert results to torch tensors where possible.
        mask_tensor = torch.from_numpy(masks[0])
        score_tensor = torch.tensor(scores[0])
        all_masks.append(mask_tensor)
        all_scores.append(score_tensor)
        processed_names.append(valid_images[idx])

        if len(all_masks) % 500 == 0:
            print(f"\nProcessed {len(all_masks)} images")
            avg_score = torch.stack(all_scores).float().mean().item()
            print(f"Current average score: {avg_score:.3f}")
            show_masks(np.array(img), masks, scores, point_coords=point_coords, input_labels=point_labels)

    except Exception as e:
        print(f"\nError processing {valid_images[idx]}: {str(e)}")

# === SAVE RESULTS USING TORCH.SAVE ===
save_path = os.path.join(output_dir, f'{save_name}.pt')
if os.path.exists(save_path):
    override = input(f"File {save_name}.pt already exists. Override? (y/n): ").lower()
    if override != 'y':
        print("Operation cancelled")
        exit()

results = {
    'masks': all_masks,                      # List of torch tensors.
    'scores': torch.stack(all_scores) if all_scores else torch.tensor([]),  # Combined tensor of scores.
    'image_names': processed_names           # List of strings.
}
torch.save(results, save_path)

avg_final_score = torch.stack(all_scores).float().mean().item() if all_scores else 0.0

print("\nProcessing complete!")
print(f"Total images processed: {len(processed_names)}")
print(f"Final average score: {avg_final_score:.3f}")
