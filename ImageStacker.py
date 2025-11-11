#!/usr/bin/env python3

# 🚀 ImageStacker.py: A Python Engine to Stack Images for Astrophotography
# Author: Timothy Do
# Updated by: Gemini (with multiprocessing, argparse, and SAM2 integration)
# v14: Implements user suggestion to allow astroalign (aa) to be used
#      with masks by pre-masking the images.

# Importing Libraries
import os
import sys  # Added for SAM2/hydra fix
import numpy as np
import cv2 as cv
import rawpy
from tqdm import tqdm
import torch
import kornia as K
import gc
from datetime import datetime
import astroalign as aa
import argparse
import glob
from multiprocessing import Pool, cpu_count
import requests  # Added for SAM2 downloader

# --- Normalization and Type Conversion Functions (Unchanged) ---
def normalize(img):
    """Normalizes a float image to the range [0, 1]."""
    min_val = img.min()
    max_val = img.max()
    if max_val == min_val:
        return np.zeros_like(img)
    return (img - min_val) / (max_val - min_val)
def contrast_stretch(img,min=0,max=1):
    """Stretches the contrast of a float image to a specified min/max range."""
    return np.clip(normalize(img) * (max-min) + min,a_min=0,a_max=1)
def int_to_float(img):
    """Converts a uint8 or uint16 image to a float image normalized to [0, 1]."""
    return normalize(img.astype(np.float32))
def float_to_int(img):
    """Converts a float image normalized to [0, 1] to a uint8 image."""
    return ((2**8-1)*normalize(img)).astype(np.uint8)
def float_to_int16(img):
    """Converts a float image normalized to [0, 1] to a uint16 image."""
    return ((2**16-1)*normalize(img)).astype(np.uint16)

# --- Raw Image Processing (Unchanged) ---
def raw_to_numpy(rawPath,uint=False,gamma=(1,1),half_size=True,use_camera_wb=False,no_auto_bright=False):
    """Converts a raw image file to a NumPy array."""
    with rawpy.imread(rawPath) as raw:
        if(uint):
            out = raw.postprocess() # uint8
        else:
            out = raw.postprocess(gamma=gamma,use_camera_wb=use_camera_wb,use_auto_wb=True,no_auto_bright=no_auto_bright,half_size=half_size,output_bps=16,demosaic_algorithm=rawpy.DemosaicAlgorithm.VNG,output_color=rawpy.ColorSpace.sRGB)
            out = out.astype(np.float32)/ 2**16
        if(out.shape[2] > out.shape[1]): 
            out = np.rot90(out,k=3)
    return out
def rawDir_to_tensor(stackPath,gamma=(1,1),use_camera_wb=False,half_size=True,no_auto_bright=False,exclude=[],verbose=False):
    """Converts a directory of raw images into a stacked NumPy array (tensor)."""
    stackFiles = sorted([file for file in os.listdir(stackPath) if (file not in exclude) and (file.endswith('.CR2') or file.endswith('.CR3') or file.endswith('.ARW') or file.endswith('.DNG'))])
    if(verbose):
        print(f'Stack Files: {stackFiles}')
        print('Stacking Raw Images into a Tensor')
    stackedNumpy = np.stack([raw_to_numpy(f'{stackPath}/{stackFile}',gamma=gamma,use_camera_wb=use_camera_wb,half_size=half_size,no_auto_bright=no_auto_bright) for stackFile in (tqdm(stackFiles) if verbose else stackFiles)])
    if(verbose):
        print(f'Stacked Numpy Shape: {stackedNumpy.shape}')
        print(f'Stacked Numpy Data Type: {stackedNumpy.dtype}')
    gc.collect()
    return stackedNumpy 

# --- Image Registration Functions (Unchanged) ---
def register_image(img1,img2,numFeatures=10000,match=0.9,refill=True, mask=None):
    """
    Registers img1 to img2 using ORB features and homography.
    The 'mask' (uint8, 255 for sky) is used to find features ONLY in the sky.
    """
    gray1 = cv.cvtColor(float_to_int(img1), cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(float_to_int(img2), cv.COLOR_BGR2GRAY)
    height, width = gray1.shape
    orb_detector = cv.ORB_create(numFeatures)
    kp1, d1 = orb_detector.detectAndCompute(gray1, mask)
    kp2, d2 = orb_detector.detectAndCompute(gray2, mask)
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        print("Warning: Not enough keypoints or descriptors found (check mask?). Returning original image (img1).")
        return img1 
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
    matches = list(matcher.match(d1, d2))
    matches.sort(key = lambda x: x.distance)
    matches = matches[:int(len(matches)*match)]
    no_of_matches = len(matches)
    if no_of_matches < 4:
        print(f"Warning: Not enough good matches ({no_of_matches}) found for homography. Returning original image (img1).")
        return img1
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt
    homography, mask_homography = cv.findHomography(p1, p2, cv.RANSAC)
    if homography is None:
        print("Warning: Could not compute homography. Returning original image (img1).")
        return img1
    transformed_img = cv.warpPerspective(img1, homography, (width, height))
    if(refill): 
        transformed_img = np.where(transformed_img != 0, transformed_img, img2) 
    return transformed_img
def register_images_by_filepath(img1,img2,img1_path, img2_path,refill=True):
    """Crude image registration based on Earth's rotation and file timestamps."""
    height, width = img1.shape[:2]
    time1 = datetime.fromtimestamp(os.path.getctime(img1_path))
    time2 = datetime.fromtimestamp(os.path.getctime(img2_path))
    time_delta_seconds = (time2 - time1).total_seconds()
    earth_rotation_rate = 2 * np.pi / (24 * 60 * 60)
    rotation_angle = earth_rotation_rate * time_delta_seconds
    rotation_matrix = cv.getRotationMatrix2D((width / 2, height / 2), np.degrees(rotation_angle), 1)
    registered_img2 = cv.warpAffine(img1, rotation_matrix, (width, height))
    if(refill): 
        registered_img2 = np.where(registered_img2 != 0, registered_img2, img1) 
    return registered_img2

# --- In-Memory Stacking Function (Unchanged) ---
def stack_images(imgStack,method='mean',align=True,refIndex=0.5,refill=True,alignMethod='orb'):
    """
    Stacks images from a pre-loaded NumPy stack.
    *** NOTE: This is the high-memory-usage version. ***
    """
    refIndex = len(imgStack) // 2 if refIndex == 0.5 else refIndex
    refImg = normalize(imgStack[refIndex])
    newImgs = []
    
    for i in tqdm(range(imgStack.shape[0]), desc="Aligning (In-Memory)"): 
        processed_img = normalize(imgStack[i])
        if i == refIndex:
            newImgs.append(processed_img)
        else:
            if align:
                if alignMethod=='orb':
                    newImgs.append(register_image(processed_img, refImg, refill=refill))
                elif alignMethod=='aa':
                    registered_img, footprint = aa.register(processed_img, refImg)
                    if refill:
                        footprint_3d = footprint[..., np.newaxis]
                        registered_img = np.where(footprint_3d > 0, registered_img, refImg)
                    newImgs.append(registered_img)
            else:
                newImgs.append(processed_img)
    
    imgStack_aligned = np.stack(newImgs)
    stackedImg = np.median(imgStack_aligned,axis=0) if method == 'med' else np.mean(imgStack_aligned,axis=0)
    return stackedImg

# --- Sky/Foreground Mask Generation (Unchanged) ---
def create_optical_flow_mask(stackFiles, rawSettings, motion_threshold=0.1, verbose=False):
    """
    Generates a mask to differentiate moving (sky/stars = 255) from static (foreground = 0)
    using optical flow.
    """
    if len(stackFiles) < 2:
        print("Not enough images to compute optical flow for mask generation.")
        return None
    prev_img_color = np.clip(raw_to_numpy(stackFiles[0], *rawSettings), a_min=0, a_max=1)
    prev_gray = cv.cvtColor(float_to_int(prev_img_color), cv.COLOR_BGR2GRAY)
    initial_mask = np.zeros_like(prev_gray, dtype=np.uint8)
    flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                       poly_n=5, poly_sigma=1.2, flags=0)
    if verbose:
        print("Generating optical flow mask...")
    for i in tqdm(range(1, len(stackFiles)), desc="Optical Flow Mask"):
        curr_img_color = np.clip(raw_to_numpy(stackFiles[i], *rawSettings), a_min=0, a_max=1)
        curr_gray = cv.cvtColor(float_to_int(curr_img_color), cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **flow_params)
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        current_moving_regions = np.where(magnitude > motion_threshold, 255, 0).astype(np.uint8)
        initial_mask = cv.bitwise_or(initial_mask, current_moving_regions)
        prev_gray = curr_gray
    kernel = np.ones((7,7), np.uint8)
    initial_mask = cv.morphologyEx(initial_mask, cv.MORPH_CLOSE, kernel, iterations=2)
    initial_mask = cv.medianBlur(initial_mask, 7) 
    _, final_mask = cv.threshold(initial_mask, 127, 255, cv.THRESH_BINARY) 
    if verbose:
        print("Optical flow mask generated.")
    return final_mask

# --- Temporal Median Reference (Unchanged) ---
def create_temporal_median_reference(stackFiles, rawSettings, num_processes, verbose=False):
    """
    Creates a star-free reference image by median stacking all frames.
    This image is used as the alignment reference, not a mask.
    """
    if not stackFiles:
        return None
    
    if verbose:
        print(f"Creating Temporal Median reference image using {num_processes} processes...")
        
    tasks = [(f, rawSettings) for f in stackFiles]
    light_stack = []
    
    # Use worker to load images in parallel (using the dark frame loading logic)
    with Pool(processes=num_processes) as pool:
        pbar = tqdm(pool.imap(load_raw_worker, tasks), total=len(stackFiles), desc="Loading for Median")
        light_stack = [img for img in pbar]
        
    if verbose:
        print("Calculating temporal median...")
        
    # Stack images and calculate the median along the time axis (axis=0)
    image_stack = np.array(light_stack, dtype=np.float32)
    median_image = np.median(image_stack, axis=0)

    # Note: We return the image, not a mask.
    return np.clip(median_image, a_min=0, a_max=1)

# --- MOG2 Mask Generation (Unchanged) ---
def create_mog2_mask(stackFiles, rawSettings, mog2_history=50, verbose=False):
    """
    Generates a mask using the MOG2 background subtractor.
    """
    if not stackFiles:
        return None

    # MOG2 is an adaptive model that needs to be trained sequentially
    backSub = cv.createBackgroundSubtractorMOG2(
        history=mog2_history, 
        varThreshold=16, 
        detectShadows=False
    )
    
    fg_mask = None
    if verbose:
        print("Generating MOG2 mask...")
        
    for i in tqdm(range(len(stackFiles)), desc="MOG2 Mask"):
        frame_color = np.clip(raw_to_numpy(stackFiles[i], *rawSettings), a_min=0, a_max=1)
        frame_uint8 = cv.cvtColor(float_to_int(frame_color), cv.COLOR_BGR2GRAY)
        # Apply the algorithm to each frame to train the model and get the current mask
        fg_mask = backSub.apply(frame_uint8) 

    # Clean up the mask using morphological operations
    if fg_mask is not None:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv.dilate(fg_mask, kernel, iterations=1) 
        fg_mask = cv.erode(fg_mask, kernel, iterations=1)
        
    if verbose:
        print(f"MOG2 mask generated using {mog2_history} frames.")
        
    return fg_mask

# --- Dark Frame Stacking (Unchanged) ---
def load_raw_worker(file_path_and_settings):
    """Helper worker to load a single raw file."""
    file_path, rawSettings = file_path_and_settings
    return raw_to_numpy(file_path, *rawSettings)
def stack_darks(dark_files, rawSettings, num_processes, verbose=False):
    """
    Loads and stacks dark frames in parallel using median stacking.
    """
    if not dark_files:
        return None
    if verbose:
        print(f"Stacking {len(dark_files)} dark frames using {num_processes} processes...")
    tasks = [(f, rawSettings) for f in dark_files]
    dark_stack = []
    with Pool(processes=num_processes) as pool:
        pbar = tqdm(pool.imap(load_raw_worker, tasks), total=len(dark_files), desc="Stacking Darks")
        dark_stack = [img for img in pbar]
    if verbose:
        print("Median stacking darks...")
    stackedDark = np.median(np.stack(dark_stack), axis=0)
    gc.collect()
    if verbose:
        print("Dark frame stacking complete.")
    return stackedDark

# --- Multiprocessing Worker Functions (MODIFIED v14) ---
worker_globals = {}
def init_worker(ref_img, align_mask, raw_settings_tpl, align_method_str, refill_bool, stacked_dark_frame):
    """
    Initializes each worker process with shared read-only data
    to avoid pickling large objects for every task.
    """
    worker_globals['ref_img'] = ref_img
    worker_globals['static_align_mask'] = align_mask # This is the uint8 mask
    worker_globals['rawSettings'] = raw_settings_tpl
    worker_globals['alignMethod'] = align_method_str
    worker_globals['refill'] = refill_bool
    worker_globals['stackedDark'] = stacked_dark_frame
    
def process_image_worker(img_path):
    """
    Worker function to load, subtract dark, and align one image.
    This function is executed in a separate process.
    
    RETURNS: A tuple (img1, img2)
    - (aligned_full_frame, None) : For full-frame alignment
    - (unaligned_original, aligned_sky_only) : For masked stacking
    """
    refImg_orig = worker_globals['ref_img']
    static_align_mask = worker_globals['static_align_mask']
    rawSettings = worker_globals['rawSettings']
    alignMethod = worker_globals['alignMethod']
    refill = worker_globals['refill']
    stackedDark = worker_globals['stackedDark']
    
    capImg_orig = np.clip(raw_to_numpy(img_path, *rawSettings) - (stackedDark if stackedDark is not None else 0), a_min=0, a_max=1)
    
    # --- (MODIFIED v14) Masked Stacking Path (for ORB and AA) ---
    if alignMethod in ['orb', 'aa'] and static_align_mask is not None:
        
        aligned_sky_portion = None
        
        if alignMethod == 'orb':
            # ORB alignment, guided by the mask
            aligned_sky_portion = register_image(capImg_orig, refImg_orig, refill=False, mask=static_align_mask)
        
        elif alignMethod == 'aa':
            # Astroalign: Pre-mask both images to align only the sky
            try:
                mask_3ch = np.stack([static_align_mask]*3, axis=-1) / 255.0
                capImg_sky_only = capImg_orig * mask_3ch
                refImg_sky_only = refImg_orig * mask_3ch
                
                # aa.register will align the skies and return an image with a black, unaligned foreground
                aligned_sky_portion, footprint = aa.register(capImg_sky_only, refImg_sky_only)
            except Exception as e:
                print(f"Warning: Astroalign failed for {img_path}. Returning original image. Error: {e}")
                aligned_sky_portion = capImg_orig * (np.stack([static_align_mask]*3, axis=-1) / 255.0) # Return unaligned sky
        
        gc.collect() 
        # Return the UNALIGNED original and the ALIGNED sky
        return capImg_orig, aligned_sky_portion 
    
    # --- Full-Frame Stacking Paths ---
    
    registered_capImg = None
    
    if alignMethod == 'orb': # mask is None
        registered_capImg = register_image(capImg_orig, refImg_orig, refill=True, mask=None)
        
    elif alignMethod == 'aa': # mask is None
        try:
            registered_img, footprint = aa.register(capImg_orig, refImg_orig)
            if refill:
                footprint_3d = footprint[..., np.newaxis]
                registered_capImg = np.where(footprint_3d > 0, registered_img, refImg_orig)
            else:
                registered_capImg = registered_img
        except Exception as e:
             print(f"Warning: Astroalign failed for {img_path}. Returning original image. Error: {e}")
             registered_capImg = capImg_orig
            
    else: # No alignment ('none')
        registered_capImg = capImg_orig
        
    gc.collect() 
    # Return the ALIGNED full-frame and None as a placeholder
    return registered_capImg, None


# ============================================
# --- SKY SEGMENTATION MODULE (SAM2) ---
# This section is adapted from SkySegmentation.py (v13)
# ============================================

def _ss_download_model_checkpoint(url: str, save_path: str):
    """Downloads a file from a URL to a save path with a progress bar."""
    print(f"SAM2 model checkpoint not found at {save_path}.")
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
        # Do not exit, just return
        
# --- MODIFIED FOR v9: Enhanced Sky Prompt ---
def _ss_get_sky_prompt(image: np.ndarray, include_negative_prompt: bool = False) -> (np.ndarray, np.ndarray):
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

# --- MODIFIED: Added half_size and include_negative_prompt parameters ---
def create_sam2_mask(image_path: str, device_choice: str, half_size: bool, include_negative_prompt: bool = False) -> np.ndarray:
    """
    Main SAM2 segmentation function, refactored to be a callable
    module function. It loads an image, segments the sky, and
    returns the mask as a uint8 NumPy array (or None on failure).
    """
    
    # --- (HYDRA FIX) Clean sys.argv *before* importing SAM2 ---
    original_argv = list(sys.argv)
    sys.argv = [original_argv[0]] # Keep only script name
    
    try:
        # --- Import SAM2 ---
        print("Importing SAM2 library for mask generation...")
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
            return None
            
        # --- Find SAM2 Repo Root ---
        try:
            sam2_pkg_file = sam2.__file__
            sam2_pkg_dir = os.path.dirname(os.path.abspath(sam2_pkg_file))
            SAM2_REPO_ROOT = os.path.dirname(sam2_pkg_dir)
            print(f"Found SAM2 repository root at: {SAM2_REPO_ROOT}")
        except Exception:
            print("Error: Could not find the SAM2 repository root from 'sam2' package.")
            return None

        # --- Define Paths ---
        MODEL_CONFIG_NAME = "sam2_hiera_l.yaml"
        MODEL_CONFIG_ABSPATH = os.path.join(sam2_pkg_dir, MODEL_CONFIG_NAME)
        CHECKPOINT_DIR = os.path.join(SAM2_REPO_ROOT, "checkpoints")
        MODEL_CHECKPOINT_NAME = "sam2_hiera_large.pt"
        MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, MODEL_CHECKPOINT_NAME)
        MODEL_CHECKPOINT_URL = "https.dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"

        # --- Start Original Logic ---
        print(f"Loading reference image for SAM2: {image_path}")
        
        # --- 1. Load Image (Robustly) ---
        try:
            image_float = raw_to_numpy(
                image_path, 
                half_size=half_size, 
                no_auto_bright=True, 
                use_camera_wb=False
            )
            
            # --- (NEW v13) Apply contrast stretch before SAM2 ---
            print("Applying contrast stretch to image before SAM2 segmentation...")
            image_stretched_float = contrast_stretch(image_float)
            # --- (End NEW v13) ---

            image_rgb = (image_stretched_float * 255).astype(np.uint8) # SAM expects uint8
            
        except Exception as e:
            print(f"Error: Failed to read file bytes from {image_path} for SAM2: {e}")
            image_rgb = None
        
        if image_rgb is None:
            print(f"Error: Could not read or decode image from {image_path} for SAM2")
            return None
            
        # --- 2. Check for Model Files ---
        if not os.path.exists(MODEL_CONFIG_ABSPATH):
            print(f"Error: SAM2 model config not found at {MODEL_CONFIG_ABSPATH}")
            return None
        
        if not os.path.exists(MODEL_CHECKPOINT_PATH):
            _ss_download_model_checkpoint(MODEL_CHECKPOINT_URL, MODEL_CHECKPOINT_PATH)

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
        
        print(f"Using SAM2 device: {device}")
        
        try:
            sam2_model = build_sam2(MODEL_CONFIG_NAME, MODEL_CHECKPOINT_PATH, device=device)
            predictor = SAM2ImagePredictor(sam2_model)
        except Exception as e:
            print(f"Error initializing SAM2 model: {e}")
            return None

        # --- 4. Set Image and Generate Prompt ---
        print("Processing image with SAM2...")
        # --- MODIFIED FOR v9: Pass include_negative_prompt ---
        point_coords, point_labels = _ss_get_sky_prompt(image_rgb, include_negative_prompt=include_negative_prompt)
        print(f"Using SAM2 point prompt: coords={point_coords}, label={point_labels}")
        
        predictor.set_image(image_rgb)

        # --- 5. Run Prediction ---
        print("Running SAM2 segmentation prediction...")
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        
        # --- 6. Select Best Mask ---
        best_mask_index = np.argmax(scores)
        best_mask = masks[best_mask_index]
        best_score = scores[best_mask_index]
        
        print(f"SAM2 segmentation complete. Best mask score: {best_score:.4f}")

        # --- 7. Return Mask ---
        # Return the mask in the format required by register_image (uint8, sky=255)
        return (best_mask * 255).astype(np.uint8)

    except Exception as e:
        print(f"An unexpected error occurred during SAM2 processing: {e}")
        return None
    finally:
        sys.argv = original_argv # Restore original argv
        
# ============================================
# --- END SAM2 MODULE ---
# ============================================


# --- REFACTORED: Optimized Stacking Function (MODIFIED FOR v14) ---

def stack_images_opt(
    stackFiles, num_processes, gamma=(1,1), use_camera_wb=False, 
    half_size=True, no_auto_bright=False, align=True, refIndex=0.5, 
    refill=True, alignMethod='orb', stackedDark=None, 
    mask_source='optical_flow', motion_threshold=0.1, mog2_history=50, 
    device_choice=None, 
    dilate_mask_pixels=0,
    feather_mask_pixels=0,
    sam2_negative_prompt=False, 
    average_foreground=False,
    verbose=False, debug=False
):
    """
    Stacks images from file paths with automatic alignment, optimized for memory
    and parallelized for speed.
    """
    rawSettings = (False,gamma,half_size,use_camera_wb,no_auto_bright)
    refIndex = len(stackFiles) // 2 if refIndex == 0.5 else int(refIndex)
    
    if refIndex < 0 or refIndex >= len(stackFiles):
        print(f"Error: refIndex {refIndex} is out of bounds. Defaulting to middle image.")
        refIndex = len(stackFiles) // 2
        
    all_files_for_mask = stackFiles[:]
    ref_path = all_files_for_mask.pop(refIndex)
    all_files_for_mask.insert(0, ref_path) 
    
    num_files = len(stackFiles)
    static_align_mask = None
    refImg_orig = None
    asset_to_save = None
    
    # --- MASK / REFERENCE IMAGE GENERATION LOGIC ---
    
    # --- (MODIFIED v14) Generate mask if align is on AND mask source is specified ---
    if align and alignMethod != 'none' and mask_source != 'none' and mask_source != 'temporal_median':
        if mask_source == 'optical_flow':
            print("Calculating optical flow mask...")
            static_align_mask = create_optical_flow_mask(all_files_for_mask, rawSettings, motion_threshold=motion_threshold, verbose=verbose)
            asset_to_save = static_align_mask 
        
        elif mask_source == 'mog2':
            print("Calculating MOG2 mask...")
            static_align_mask = create_mog2_mask(all_files_for_mask, rawSettings, mog2_history=mog2_history, verbose=verbose)
            asset_to_save = static_align_mask
        
        elif mask_source == 'sam2':
            print("Calculating SAM2 sky mask... (This may take a while the first time)")
            static_align_mask = create_sam2_mask(
                ref_path, 
                device_choice=device_choice, 
                half_size=half_size,
                include_negative_prompt=sam2_negative_prompt
            ) 
            asset_to_save = static_align_mask
            if static_align_mask is None:
                print("Warning: SAM2 mask generation failed.")
    
    # --- (MODIFIED v14) Reference Image Logic ---
    if mask_source == 'temporal_median':
        print("Creating Temporal Median reference image.")
        refImg_orig = create_temporal_median_reference(all_files_for_mask, rawSettings, num_processes, verbose=verbose)
        asset_to_save = refImg_orig
        print("Temporal Median reference created. Mask is not used in this mode.")
        static_align_mask = None # Explicitly disable mask
    else:
        # Default reference image if not created by temporal median
        refImg_orig = np.clip(raw_to_numpy(ref_path, *rawSettings) - (stackedDark if stackedDark is not None else 0),a_min=0,a_max=1)

    if static_align_mask is None and mask_source not in ['none', 'temporal_median']:
        print(f"Warning: {mask_source} mask failed. Falling back to full-frame alignment.")
            
    if refImg_orig is None:
        raise ValueError("Could not establish a reference image for stacking.")

    # --- EXECUTION ---
    
    stackFiles_to_process = all_files_for_mask[1:] 
    
    # --- (MODIFIED v14) Masked stacking now works for orb AND aa ---
    masked_stacking = (align and alignMethod in ['orb', 'aa'] and static_align_mask is not None)
    
    init_args = (refImg_orig, static_align_mask, rawSettings, (alignMethod if align else 'none'), refill, stackedDark)
    
    if verbose:
        print(f"Starting stacking pool with {num_processes} processes...")
        if masked_stacking:
            print(f"Using masked stacking (sky/foreground separation) with '{mask_source}' and '{alignMethod}'.")
            if average_foreground:
                print("Foreground averaging is ON (noise reduction).")
            else:
                print("Foreground averaging is OFF (using sharp reference frame).")
        else:
            print(f"Using full-frame alignment with '{alignMethod}'.")

    if masked_stacking:
        
        if static_align_mask.shape[:2] != refImg_orig.shape[:2]:
             print(f"Warning: Mask dimensions {static_align_mask.shape[:2]} do not match reference frame {refImg_orig.shape[:2]}.")
             print("Upsampling mask to match reference frame...")
             static_align_mask = cv.resize(
                 static_align_mask, 
                 (refImg_orig.shape[1], refImg_orig.shape[0]), # (width, height)
                 interpolation=cv.INTER_NEAREST
             )
        
        if dilate_mask_pixels > 0:
            print(f"Dilating sky mask by {dilate_mask_pixels} pixels to shrink foreground...")
            ksize = int(dilate_mask_pixels) * 2 + 1
            kernel = np.ones((ksize, ksize), np.uint8)
            static_align_mask = cv.dilate(static_align_mask, kernel, iterations=1)
            asset_to_save = static_align_mask
            
        if feather_mask_pixels > 0:
            print(f"Feathering mask by {feather_mask_pixels} pixels...")
            ksize = int(feather_mask_pixels) * 2 + 1
            static_align_mask = cv.GaussianBlur(static_align_mask, (ksize, ksize), 0)
            asset_to_save = static_align_mask
        
        mask_3ch = np.stack([static_align_mask]*3, axis=-1) / 255.0
        inv_mask_3ch = 1.0 - mask_3ch
        
        stacked_sky = (refImg_orig * mask_3ch) / num_files
        
        if average_foreground:
            stacked_foreground = (refImg_orig * inv_mask_3ch) / num_files
        else:
            stacked_foreground = (refImg_orig * inv_mask_3ch)
        
        with Pool(processes=num_processes, initializer=init_worker, initargs=init_args) as pool:
            tasks = stackFiles_to_process
            pbar = tqdm(pool.imap_unordered(process_image_worker, tasks), total=len(tasks), desc="Aligning Sky")
            
            for i, (unaligned_orig, aligned_sky) in enumerate(pbar):
                if unaligned_orig is None or aligned_sky is None:
                     print(f"Warning: Worker returned None for image {i}. Skipping.")
                     continue
                     
                stacked_sky += (aligned_sky * mask_3ch) / num_files
                
                if average_foreground:
                    stacked_foreground += (unaligned_orig * inv_mask_3ch) / num_files
                
                if verbose and ((i+1) % 20 == 0): 
                    print(f"  Processed {i+1}/{len(tasks)} images...")
        
        if debug:
            try:
                debug_dir = "debug"
                os.makedirs(debug_dir, exist_ok=True)
                print(f"Debug mode: Saving intermediate stacks to '{debug_dir}/'...")
                
                sky_stretched = contrast_stretch(stacked_sky)
                sky_16bit = float_to_int16(sky_stretched)
                if sky_16bit.shape[2] == 3: 
                    sky_16bit = sky_16bit[...,::-1]
                cv.imwrite(os.path.join(debug_dir, "debug_stacked_sky.tiff"), sky_16bit)
                
                fg_stretched = contrast_stretch(stacked_foreground)
                fg_16bit = float_to_int16(fg_stretched)
                if fg_16bit.shape[2] == 3: 
                    fg_16bit = fg_16bit[...,::-1]
                cv.imwrite(os.path.join(debug_dir, "debug_stacked_foreground.tiff"), fg_16bit)
                
            except Exception as e:
                print(f"Error during debug save: {e}")
                
        print("Merging stacked sky and foreground...")
        stackedImg = stacked_sky + stacked_foreground
        
    else: # Full-frame stacking
        stackedImg = refImg_orig / num_files
        
        with Pool(processes=num_processes, initializer=init_worker, initargs=init_args) as pool:
            tasks = stackFiles_to_process
            pbar = tqdm(pool.imap_unordered(process_image_worker, tasks), total=len(tasks), desc="Aligning & Stacking")
            
            for i, (aligned_img, _) in enumerate(pbar): 
                if aligned_img is None:
                     print(f"Warning: Worker returned None for image {i}. Skipping.")
                     continue
                
                stackedImg += aligned_img / num_files
                if verbose and ((i+1) % 20 == 0): 
                    print(f"  Processed {i+1}/{len(tasks)} images...")

    gc.collect()
    print("Stacking complete.")
    
    return stackedImg, asset_to_save

# --- Crude Stacking (Unchanged) ---
def stack_images_crude(stackFiles,gamma=(1,1),use_camera_wb=False,half_size=True,no_auto_bright=False,refIndex=0.5,align=False,stackedDark=None):
    """
    Stacks images using crude alignment based on Earth's rotation.
    """
    stackedImg = 0
    rawSettings = (False,gamma,half_size,use_camera_wb,no_auto_bright)
    refIndex = len(stackFiles) // 2 if refIndex == 0.5 else int(refIndex)
    refImg = np.clip(raw_to_numpy(stackFiles[refIndex],*rawSettings) - (stackedDark if stackedDark is not None else 0),a_min=0,a_max=1)
    refPath = stackFiles[refIndex]
    for i in tqdm(range(len(stackFiles)), desc="Crude Stacking"):
        if i == refIndex:
            stackedImg += refImg/len(stackFiles) 
        else: 
            capImg = np.clip(raw_to_numpy(stackFiles[i],*rawSettings) - (stackedDark if stackedDark is not None else 0),a_min=0,a_max=1)
            stackedImg += (register_images_by_filepath(refImg,capImg,refPath,stackFiles[i]) if align else capImg)/len(stackFiles)
        gc.collect()
    return stackedImg


# --- Main execution block with argparse (MODIFIED FOR v14) ---

def main():
    parser = argparse.ArgumentParser(description="🚀 ImageStacker.py: Stack raw astrophotography images.")
    
    # --- File I/O Arguments ---
    parser.add_argument('--lights-dir', type=str, required=True,
                        help="Directory containing the light frames (raw images).")
    parser.add_argument('--darks-dir', type=str, default=None,
                        help="Directory containing the dark frames (optional).")
    parser.add_argument('--output-file', type=str, default='./Stacked_Image.tiff',
                        help="Path to save the final stacked image (e.g., stacked.tiff).")
    parser.add_argument('--save-mask', type=str, default=None,
                        help="Path to save the generated asset (e.g., mask.jpg or reference.tiff).")
    parser.add_argument('--glob-pattern', type=str, default='*.CR[23]',
                        help="Glob pattern to find raw files (e.g., '*.ARW', '*.CR2', '*.dng').")
                        
    # --- Stacking & Alignment Arguments ---
    parser.add_argument('-j', '--processes', type=int, default=cpu_count(),
                        help=f"Number of processes to use. Default: all available ({cpu_count()}).")
    parser.add_argument('--align', action='store_true',
                        help="Enable image alignment. (Default: False)")
    parser.add_argument('--align-method', type=str, default='orb', choices=['orb', 'aa', 'none'],
                        help="Alignment method: 'orb' (OpenCV, mask-aware), 'aa' (astroalign, now mask-aware).")
    parser.add_argument('--ref-index', type=float, default=0.5,
                        help="Reference image index. 0.5 for middle (default), or a specific index (e.g., 0).")
    
    # (v10) Foreground Averaging Toggle
    parser.add_argument(
        "--average-foreground",
        action="store_true",
        help="Average the unaligned foregrounds to reduce noise. "
             "Default is to use the single, sharp foreground from the reference frame."
    )
                        
    # --- MASK SOURCE ARGUMENT (MODIFIED) ---
    parser.add_argument('--mask-source', type=str, default='optical_flow', 
                        choices=['optical_flow', 'temporal_median', 'mog2', 'sam2', 'none'],
                        help="Source for the sky/foreground separation. 'sam2' uses Segment Anything 2. 'temporal_median' and 'none' disable the mask.")

    # --- MASK PARAMETER ARGUMENTS (MODIFIED) ---
    parser.add_argument('--motion-threshold', type=float, default=0.1,
                        help="Optical flow motion threshold for sky/foreground mask. (Used with --mask-source optical_flow)")
    parser.add_argument('--mog2-history', type=int, default=50,
                        help="Number of frames used to build the background model for MOG2. (Used with --mask-source mog2)")
    
    # (v12) Changed --erode-mask to --dilate-mask
    parser.add_argument(
        "--dilate-mask",
        type=int,
        default=0,
        help="The number of pixels to dilate the sky mask (shrinks foreground) to make it tighter. "
             "Applied before feathering. Try a value like 10 or 15. (Default: 0)"
    )
    
    # (v7) Feathering argument
    parser.add_argument(
        "--feather-mask",
        type=int,
        default=0,
        help="The number of pixels to blur/feather the mask edge for a smoother blend. "
             "Applied after dilation. Try a value like 10 or 15. (Default: 0, a hard edge)"
    )
    
    # (v9) SAM2 Negative Prompt Argument
    parser.add_argument(
        "--sam2-negative-prompt",
        action="store_true",
        help="Include a negative prompt point for SAM2 in the bottom-center to help delineate foreground. "
             "Only used when --mask-source sam2."
    )
                        
    # --- SAM2 Device Argument ---
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=['cuda', 'mps', 'cpu'],
        help="Manually specify the compute device for SAM2 ('cuda', 'mps', 'cpu'). "
             "Default: auto-detect (cuda > mps > cpu)."
    )
    
    # --- DEBUG Argument ---
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode. Saves intermediate stacks (sky, foreground) to a 'debug' folder."
    )
                        
    # --- Raw Processing Arguments (Unchanged) ---
    parser.add_argument('--half-size', action='store_true',
                        help="Process raw files at half resolution for speed.")
    parser.add_argument('--no-auto-bright', action='store_true',
                        help="Disable rawpy's automatic brightness adjustment.")
    
    # --- (FIX v8) Corrected typo 'stror_true' to 'store_true' ---
    parser.add_argument('--use-camera-wb', action='store_true',
                        help="Use the in-camera white balance setting.")

    args = parser.parse_args()

    # --- 1. Find Image Files (Unchanged) ---
    light_pattern = os.path.join(args.lights_dir, args.glob_pattern)
    stackLightFiles = sorted(glob.glob(light_pattern))
    
    if not stackLightFiles:
        print(f"Error: No light frames found in {args.lights_dir} with pattern {args.glob_pattern}")
        return

    print(f"Found {len(stackLightFiles)} light frames.")
    
    dark_files = []
    if args.darks_dir:
        dark_pattern = os.path.join(args.darks_dir, args.glob_pattern)
        dark_files = sorted(glob.glob(dark_pattern))
        if not dark_files:
            print(f"Warning: --darks-dir specified, but no darks found in {args.darks_dir}")
        else:
            print(f"Found {len(dark_files)} dark frames.")

    rawSettings = (False, (1,1), args.half_size, args.use_camera_wb, args.no_auto_bright)

    stackedDark = None
    if dark_files:
        stackedDark = stack_darks(dark_files, rawSettings, args.processes, verbose=True)

    # --- 4. Stack Light Frames (MODIFIED) ---
    print(f"Starting light frame stacking using '{args.align_method}' alignment...")
    stackedImg, mask_or_ref = stack_images_opt(
        stackLightFiles,
        num_processes=args.processes,
        gamma=(1,1),
        use_camera_wb=args.use_camera_wb,
        half_size=args.half_size,
        no_auto_bright=args.no_auto_bright,
        align=(args.align or args.align_method != 'none'),
        alignMethod=args.align_method,
        refIndex=args.ref_index,
        stackedDark=stackedDark,
        mask_source=args.mask_source, 
        motion_threshold=args.motion_threshold,
        mog2_history=args.mog2_history,
        device_choice=args.device,
        dilate_mask_pixels=args.dilate_mask,
        feather_mask_pixels=args.feather_mask,
        sam2_negative_prompt=args.sam2_negative_prompt,
        average_foreground=args.average_foreground,
        verbose=True,
        debug=args.debug
    )

    # --- 5. Save Results (Unchanged) ---
    print(f"Saving final stacked image to {args.output_file}...")
    stretched_img = contrast_stretch(stackedImg)
    output_img_16bit = float_to_int16(stretched_img)
    
    if output_img_16bit.shape[2] == 3:
         output_img_16bit = output_img_16bit[...,::-1] 
         
    cv.imwrite(args.output_file, output_img_16bit)

    # --- MODIFIED: Save Mask or Reference Image ---
    if args.save_mask and mask_or_ref is not None:
        if mask_or_ref.ndim == 3 and mask_or_ref.dtype == np.float32:
            print(f"Saving temporal median reference image to {args.save_mask}...")
            stretched_ref = contrast_stretch(mask_or_ref)
            output_ref_16bit = float_to_int16(stretched_ref)
            if output_ref_16bit.shape[2] == 3:
                output_ref_16bit = output_ref_16bit[...,::-1] 
            cv.imwrite(args.save_mask, output_ref_16bit)
        else:
            print(f"Saving alignment mask to {args.save_mask}...")
            cv.imwrite(args.save_mask, mask_or_ref)
            
    elif args.save_mask and mask_or_ref is None and args.mask_source not in ['none']:
        print(f"Note: --save-mask was specified, but the '{args.mask_source}' asset generation failed or was disabled.")
        
    print("✅ Image Stacker Done.")


if __name__ == '__main__':
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass 
        
    main()