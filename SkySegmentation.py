#!/usr/bin/env python3

"""
segment_sky_sam2.py (v15)

Description:
This script is fully standalone and can be saved ANYWHERE on your system.
It points to the correct 'facebookresearch/sam2' repository.

**This version (v15) adds support for reading RAW image files (e.g., .CR2,
.ARW, .DNG) using the 'rawpy' library and uses the advanced
multi-point prompt for sharper masks.**

It retains all previous fixes (hydra, config paths, etc.).

Author: Your Name
Date: 2025-11-09

Requirements:
- Python 3.10+
- PyTorch
- 'requests' library: pip install requests
- 'tqdm' library: pip install tqdm
- 'rawpy' library: pip install rawpy
- 'opencv-python': pip install opencv-python-headless
- 'matplotlib': pip install matplotlib

Setup:
1.  Clone the *correct* SAM2 repo:
    git clone https://github.com/facebookresearch/sam2.git
    
2.  Install the 'sam2' library in your environment:
    cd sam2
    pip install -e .
    
3.  Install other dependencies:
    pip install opencv-python-headless matplotlib requests tqdm rawpy

Usage:
# You can now pass a path to a JPG, PNG, or RAW file
python /path/to/SkySegmentation.py /path/to/image.CR2 --save --sam2-negative-prompt
"""

import argparse
import os
import sys  # Must be imported at the top
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm

try:
    import rawpy
except ImportError:
    print("Error: 'rawpy' library not found.")
    print("Please install it to read RAW files: pip install rawpy")
    rawpy = None

# --- Defer SAM2 imports ---
# We will import sam2 *inside* the main function,
# after cleaning sys.argv to prevent hydra conflicts.

# ==================================
# Helper Functions
# (These cannot depend on sam2)
# ==================================

def download_model_checkpoint(url: str, save_path: str):
    """Downloads a file from a URL to a save path with a progress bar."""
    print(f"Model checkpoint not found at {save_path}.")
    print(f"Downloading from {url}...")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(save_path, 'wb') as f, tqdm(
                desc=os.path.basename(save_path),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
                    
        print(f"Model downloaded successfully and saved to {save_path}")
    except Exception as e:
        print(f"\nError: Failed to download model checkpoint: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        sys.exit(1)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])  # Blueish
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)

def get_sky_prompt(image: np.ndarray, include_negative_prompt: bool = False) -> (np.ndarray, np.ndarray):
    """
    Generates heuristic point prompts for segmenting the sky.
    Uses multiple positive points and an optional negative point.
    """
    h, w, _ = image.shape
    
    # Positive points (label 1 for 'sky')
    # Try multiple points across the top to give SAM2 a stronger signal for the sky area
    positive_coords = np.array([
        [w // 4, h // 8],        # Top-left quadrant
        [w // 2, h // 8],        # Top-center
        [w * 3 // 4, h // 8],    # Top-right quadrant
        [w // 2, h // 4]         # Slightly lower center
    ])
    positive_labels = np.array([1, 1, 1, 1])

    point_coords = positive_coords
    point_labels = positive_labels

    # Optional: Add a negative point to guide SAM2 away from foreground if it mistakenly includes it
    if include_negative_prompt:
        # Heuristic for foreground: bottom-center
        negative_coord = np.array([[w // 2, h * 7 // 8]])
        negative_label = np.array([0]) # Label 0 for 'not sky' (foreground)
        
        point_coords = np.concatenate((point_coords, negative_coord))
        point_labels = np.concatenate((point_labels, negative_label))
        
    return point_coords, point_labels

# --- (NEW v15) ---
def read_standard_image(image_path: str) -> np.ndarray:
    """Reads a standard image (JPG, PNG) and returns an RGB uint8 array."""
    try:
        f = open(image_path, "rb")
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        f.close()
        if image_bgr is None:
            return None
        # Convert BGR (OpenCV default) to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_rgb
    except Exception as e:
        print(f"Error reading standard image {image_path}: {e}")
        return None

def read_raw_image(image_path: str) -> np.ndarray:
    """Reads a RAW image and returns an RGB uint8 array."""
    if rawpy is None:
        print("Error: 'rawpy' is required to read this file but is not installed.")
        return None
    try:
        with rawpy.imread(image_path) as raw:
            # Post-process to an 8-bit RGB image.
            # use_camera_wb=True and no_auto_bright=True are good defaults for astrophotos
            image_rgb = raw.postprocess(
                use_camera_wb=True, 
                no_auto_bright=True, 
                output_bps=8
            )
            return image_rgb
    except Exception as e:
        print(f"Error reading RAW image {image_path}: {e}")
        return None

def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from a path, automatically handling RAW and standard formats.
    Returns an RGB uint8 NumPy array or None on failure.
    """
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        return None
        
    # Define common RAW file extensions
    RAW_EXTENSIONS = ['.cr2', '.cr3', '.arw', '.dng', '.nef', '.raf', '.orf', '.rw2']
    
    try:
        ext = os.path.splitext(image_path)[1].lower()
        
        if ext in RAW_EXTENSIONS:
            print("Detected RAW file, using rawpy...")
            return read_raw_image(image_path)
        else:
            print("Detected standard image, using OpenCV...")
            return read_standard_image(image_path)
    except Exception as e:
        print(f"An error occurred during image loading: {e}")
        return None

# ==================================
# Main Segmentation Function
# ==================================

def segment_sky(image_path: str, save_mask: bool, output_path: str, device_choice: str, sam2_negative_prompt: bool) -> None:
    
    # --- (FIX v8) Clean sys.argv *before* importing SAM2 ---
    original_argv = list(sys.argv)
    sys.argv = [original_argv[0]] # Keep only script name
    
    try:
        # --- Import SAM2 ---
        print("Importing SAM2 library...")
        try:
            global sam2, build_sam2, SAM2ImagePredictor
            import sam2
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            print("---" * 20)
            print("Error: Could not import the 'sam2' library.")
            print("Please ensure you have cloned the 'facebookresearch/sam2' repository")
            print("and installed it in your environment by running:")
            print("\n  cd /path/to/sam2")
            print("  pip install -e .\n")
            print("---" * 20)
            return
            
        # --- Find SAM2 Repo Root ---
        try:
            sam2_pkg_file = sam2.__file__
            sam2_pkg_dir = os.path.dirname(os.path.abspath(sam2_pkg_file))
            SAM2_REPO_ROOT = os.path.dirname(sam2_pkg_dir)
            print(f"Found SAM2 repository root at: {SAM2_REPO_ROOT}")
        except Exception:
            print("Error: Could not find the SAM2 repository root from 'sam2' package.")
            return

        # --- Define Paths ---
        MODEL_CONFIG_NAME = "sam2_hiera_l.yaml"
        MODEL_CONFIG_ABSPATH = os.path.join(sam2_pkg_dir, MODEL_CONFIG_NAME)
        CHECKPOINT_DIR = os.path.join(SAM2_REPO_ROOT, "checkpoints")
        MODEL_CHECKPOINT_NAME = "sam2_hiera_large.pt"
        MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, MODEL_CHECKPOINT_NAME)
        MODEL_CHECKPOINT_URL = "https.dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"

        # --- Start Original Logic ---
        print(f"Loading image from: {image_path}")
        
        # --- 1. Load Image (MODIFIED v15) ---
        image_rgb = load_image(image_path)
        
        if image_rgb is None:
            print(f"Error: Could not read or decode image from {image_path}")
            return
            
        # --- 2. Check for Model Files ---
        if not os.path.exists(MODEL_CONFIG_ABSPATH):
            print(f"Error: Model config not found at {MODEL_CONFIG_ABSPATH}")
            return
        
        if not os.path.exists(MODEL_CHECKPOINT_PATH):
            download_model_checkpoint(MODEL_CHECKPOINT_URL, MODEL_CHECKPOINT_PATH)

        # --- 3. Initialize Model ---
        print("Loading SAM2 model... (This may take a moment)")
        
        device = None
        if device_choice:
            device = device_choice
        else:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        print(f"Using device: {device}")
        
        try:
            sam2_model = build_sam2(MODEL_CONFIG_NAME, MODEL_CHECKPOINT_PATH, device=device)
            predictor = SAM2ImagePredictor(sam2_model)
        except Exception as e:
            print(f"Error initializing SAM2 model: {e}")
            return

        # --- 4. Set Image and Generate Prompt ---
        print("Processing image and setting up predictor...")
        predictor.set_image(image_rgb)
        
        point_coords, point_labels = get_sky_prompt(image_rgb, include_negative_prompt=sam2_negative_prompt)
        print(f"Using point prompt: coords={point_coords}, label={point_labels}")

        # --- 5. Run Prediction ---
        print("Running segmentation prediction...")
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        
        # --- 6. Select Best Mask ---
        best_mask_index = np.argmax(scores)
        best_mask = masks[best_mask_index]
        best_score = scores[best_mask_index]
        
        print(f"Segmentation complete. Best mask score: {best_score:.4f}")

        # --- 7. Save or Display ---
        if save_mask:
            if output_path is None:
                current_dir = os.getcwd()
                results_dir = os.path.join(current_dir, "results")
                os.makedirs(results_dir, exist_ok=True)
                base = os.path.basename(image_path)
                name, ext = os.path.splitext(base)
                new_filename = f"{name}_mask.png"
                output_path = os.path.join(results_dir, new_filename)
                
            print(f"Saving binary mask to: {output_path}")
            
            mask_to_save = (best_mask * 255).astype(np.uint8)
            
            try:
                cv2.imwrite(output_path, mask_to_save)
                print("Mask saved successfully.")
            except Exception as e:
                print(f"Error saving mask: {e}")
                
        else:
            print("Displaying segmentation result. Close the window to exit.")
            plt.figure(figsize=(10, 10))
            plt.imshow(image_rgb)
            show_mask(best_mask, plt.gca())
            show_points(point_coords, point_labels, plt.gca())
            plt.title(f"SAM2 Sky Segmentation (Score: {best_score:.4f})")
            plt.axis('off')
            plt.show()

    finally:
        sys.argv = original_argv # Restore original argv
        
# ==================================
# Command-Line Interface
# ==================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment the sky from an image using SAM2.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
This script is location-independent.

Example usage:
  # Auto-detect device and save to ./results (JPG)
  python SkySegmentation.py /path/to/image.jpg --save
  
  # Auto-detect device and save to ./results (RAW)
  python SkySegmentation.py /path/to/image.CR2 --save
  
  # Force use of MPS (Apple Silicon)
  python SkySegmentation.py /path/to/image.ARW --save --device mps
  
  # Use a negative prompt to help define foreground
  python SkySegmentation.py /path/to/image.DNG --save --sam2-negative-prompt
"""
    )
    
    parser.add_argument("image_path", type=str, help="The file path to the input image (JPG, PNG, CR2, ARW, etc.).")
    
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the resulting mask to a file. "
             "Saves to './results/imagename_mask.png' in the current directory."
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Specify a custom file path to save the output mask. "
             "Overrides the default behavior."
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=['cuda', 'mps', 'cpu'],
        help="Manually specify the compute device ('cuda', 'mps', 'cpu'). "
             "Default: auto-detect (cuda > mps > cpu)."
    )
    
    parser.add_argument(
        "--sam2-negative-prompt",
        action="store_true",
        help="Include a negative prompt point in the bottom-center to help delineate foreground."
    )
    
    args = parser.parse_args()
    
    segment_sky(args.image_path, args.save, args.output, args.device, args.sam2_negative_prompt)