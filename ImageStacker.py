#!/usr/bin/env python3

# 🚀 ImageStacker.py: A Python Engine to Stack Images for Astrophotography
# Author: Timothy Do
# Updated by: Gemini (with multiprocessing, argparse, and improved mask logic)

# Importing Libraries
import os
import numpy as np
import cv2 as cv
import rawpy
from tqdm import tqdm
import torch # Not directly used, but kept
import kornia as K # Not directly used, but kept
import gc
from datetime import datetime
import astroalign as aa
import argparse
import glob
from multiprocessing import Pool, cpu_count

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

# --- Multiprocessing Worker Functions (Unchanged) ---
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
    - (aligned_full_frame, None) : For full-frame alignment (aa, orb-no-mask, none)
    - (unaligned_original, aligned_sky_only) : For masked ORB alignment
    """
    refImg_orig = worker_globals['ref_img']
    static_align_mask = worker_globals['static_align_mask']
    rawSettings = worker_globals['rawSettings']
    alignMethod = worker_globals['alignMethod']
    refill = worker_globals['refill']
    stackedDark = worker_globals['stackedDark']
    
    capImg_orig = np.clip(raw_to_numpy(img_path, *rawSettings) - (stackedDark if stackedDark is not None else 0), a_min=0, a_max=1)
    
    # --- Masked Stacking Path ---
    # This is the special case for sky/foreground separation
    if alignMethod == 'orb' and static_align_mask is not None:
        # Align using the mask to constrain feature detection to the sky
        # refill=False ensures foreground is black (or warped/black)
        aligned_sky_portion = register_image(capImg_orig, refImg_orig, refill=False, mask=static_align_mask)
        
        gc.collect() 
        # Return the UNALIGNED original and the ALIGNED sky
        return capImg_orig, aligned_sky_portion 
    
    # --- Full-Frame Stacking Paths ---
    
    registered_capImg = None
    
    if alignMethod == 'orb': # but static_align_mask is None
        # Fallback to full-frame ORB (e.g., mask_source='none' or mask failed)
        registered_capImg = register_image(capImg_orig, refImg_orig, refill=True, mask=None)
        
    elif alignMethod == 'aa':
        # Astroalign is full-frame alignment
        registered_img, footprint = aa.register(capImg_orig, refImg_orig)
        if refill:
            footprint_3d = footprint[..., np.newaxis]
            registered_capImg = np.where(footprint_3d > 0, registered_img, refImg_orig)
        else:
            registered_capImg = registered_img
            
    else: # No alignment ('none')
        registered_capImg = capImg_orig
        
    gc.collect() 
    # Return the ALIGNED full-frame and None as a placeholder
    return registered_capImg, None

# --- REFACTORED: Optimized Stacking Function (MODIFIED) ---

def stack_images_opt(stackFiles, num_processes, gamma=(1,1), use_camera_wb=False, half_size=True, no_auto_bright=False, align=True, refIndex=0.5, refill=True, alignMethod='orb', stackedDark=None, mask_source='optical_flow', motion_threshold=0.1, mog2_history=50, verbose=False):
    """
    Stacks images from file paths with automatic alignment, optimized for memory
    and parallelized for speed.
    
    Returns:
    - stackedImg (numpy.ndarray): The final stacked image.
    - asset_to_save (numpy.ndarray or None): The generated mask (uint8) or reference image (float32).
    """
    rawSettings = (False,gamma,half_size,use_camera_wb,no_auto_bright)
    refIndex = len(stackFiles) // 2 if refIndex == 0.5 else int(refIndex)
    
    if refIndex < 0 or refIndex >= len(stackFiles):
        print(f"Error: refIndex {refIndex} is out of bounds. Defaulting to middle image.")
        refIndex = len(stackFiles) // 2
        
    # Create the list of files to process (reference image needs to be the first element)
    all_files_for_mask = stackFiles[:]
    ref_path = all_files_for_mask.pop(refIndex)
    all_files_for_mask.insert(0, ref_path) # Put ref image at the start of the list
    
    num_files = len(stackFiles)

    static_align_mask = None
    refImg_orig = None
    
    # --- NEW: Variable to hold the asset (mask or ref image) for saving ---
    asset_to_save = None
    
    # --- MASK / REFERENCE IMAGE GENERATION LOGIC ---
    if align and alignMethod == 'orb':
        if mask_source == 'optical_flow':
            print("Calculating optical flow mask...")
            static_align_mask = create_optical_flow_mask(all_files_for_mask, rawSettings, motion_threshold=motion_threshold, verbose=verbose)
            asset_to_save = static_align_mask # Store the mask for saving
        elif mask_source == 'mog2':
            print("Calculating MOG2 mask...")
            static_align_mask = create_mog2_mask(all_files_for_mask, rawSettings, mog2_history=mog2_history, verbose=verbose)
            asset_to_save = static_align_mask # Store the mask for saving
        elif mask_source == 'temporal_median':
            print("Creating Temporal Median reference image.")
            refImg_orig = create_temporal_median_reference(all_files_for_mask, rawSettings, num_processes, verbose=verbose)
            asset_to_save = refImg_orig # Store the reference image for saving
            print("Temporal Median reference created. ORB mask is not used in this mode.")
            static_align_mask = None # Mask is explicitly disabled when using temporal median reference
        elif mask_source == 'none':
            print("Mask creation disabled. Aligning using full frame features.")
            static_align_mask = None

        if refImg_orig is None:
            # Default reference image if not created by temporal median
            refImg_orig = np.clip(raw_to_numpy(ref_path, *rawSettings) - (stackedDark if stackedDark is not None else 0),a_min=0,a_max=1)
            
        if static_align_mask is None and mask_source in ['optical_flow', 'mog2']:
            print(f"Warning: {mask_source} mask failed. Falling back to full-frame ORB alignment.")
            
    elif align and alignMethod == 'aa':
         print("Using 'astroalign'. Mask is ignored. Aligning based on full frame.")
         refImg_orig = np.clip(raw_to_numpy(ref_path, *rawSettings) - (stackedDark if stackedDark is not None else 0),a_min=0,a_max=1)
         
    else: # No alignment or alignMethod='none'
        print("No alignment performed.")
        refImg_orig = np.clip(raw_to_numpy(ref_path, *rawSettings) - (stackedDark if stackedDark is not None else 0),a_min=0,a_max=1)
        
    if refImg_orig is None:
        raise ValueError("Could not establish a reference image for stacking.")

    # --- EXECUTION ---
    
    # Files to be aligned and stacked (all except the reference)
    stackFiles_to_process = all_files_for_mask[1:] 
    
    # Check if we are in masked stacking mode
    masked_stacking = (align and alignMethod == 'orb' and static_align_mask is not None)
    
    init_args = (refImg_orig, static_align_mask, rawSettings, (alignMethod if align else 'none'), refill, stackedDark)
    
    if verbose:
        print(f"Starting stacking pool with {num_processes} processes...")
        if masked_stacking:
            print("Using masked stacking (sky/foreground separation).")
        else:
            print("Using full-frame stacking.")

    if masked_stacking:
        mask_3ch = np.stack([static_align_mask]*3, axis=-1) / 255.0
        inv_mask_3ch = 1.0 - mask_3ch
        
        # Initialize two accumulators
        stacked_sky = (refImg_orig * mask_3ch) / num_files
        stacked_foreground = (refImg_orig * inv_mask_3ch) / num_files
        
        with Pool(processes=num_processes, initializer=init_worker, initargs=init_args) as pool:
            tasks = stackFiles_to_process
            pbar = tqdm(pool.imap_unordered(process_image_worker, tasks), total=len(tasks), desc="Aligning Sky")
            
            for i, (unaligned_orig, aligned_sky) in enumerate(pbar):
                if unaligned_orig is None or aligned_sky is None:
                     print(f"Warning: Worker returned None for image {i}. Skipping.")
                     continue
                     
                stacked_foreground += (unaligned_orig * inv_mask_3ch) / num_files
                stacked_sky += aligned_sky / num_files 
                
                if verbose and ((i+1) % 20 == 0): 
                    print(f"  Processed {i+1}/{len(tasks)} images...")
        
        print("Merging stacked sky and foreground...")
        stackedImg = stacked_sky + stacked_foreground
        
    else: # Full-frame stacking (aa, orb-no-mask, temporal_median, none)
        stackedImg = refImg_orig / num_files
        
        with Pool(processes=num_processes, initializer=init_worker, initargs=init_args) as pool:
            tasks = stackFiles_to_process
            pbar = tqdm(pool.imap_unordered(process_image_worker, tasks), total=len(tasks), desc="Aligning & Stacking")
            
            for i, (aligned_img, _) in enumerate(pbar): # Unpack tuple, ignore second element
                if aligned_img is None:
                     print(f"Warning: Worker returned None for image {i}. Skipping.")
                     continue
                
                stackedImg += aligned_img / num_files
                if verbose and ((i+1) % 20 == 0): 
                    print(f"  Processed {i+1}/{len(tasks)} images...")

    gc.collect()
    print("Stacking complete.")
    
    # Return the stacked image and the asset (mask or ref image)
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


# --- Main execution block with argparse (MODIFIED) ---

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
                        # MODIFIED help text
                        help="Path to save the generated asset (e.g., mask.jpg or reference.tiff).")
    parser.add_argument('--glob-pattern', type=str, default='*.CR[23]',
                        help="Glob pattern to find raw files (e.g., '*.ARW', '*.CR2', '*.dng').")
                        
    # --- Stacking & Alignment Arguments ---
    parser.add_argument('-j', '--processes', type=int, default=cpu_count(),
                        help=f"Number of processes to use. Default: all available ({cpu_count()}).")
    parser.add_argument('--align', action='store_true',
                        help="Enable image alignment. (Default: False)")
    parser.add_argument('--align-method', type=str, default='orb', choices=['orb', 'aa', 'none'],
                        help="Alignment method: 'orb' (OpenCV, prefers mask), 'aa' (astroalign).")
    parser.add_argument('--ref-index', type=float, default=0.5,
                        help="Reference image index. 0.5 for middle (default), or a specific index (e.g., 0).")
                        
    # --- MASK SOURCE ARGUMENT (Unchanged) ---
    parser.add_argument('--mask-source', type=str, default='optical_flow', 
                        choices=['optical_flow', 'temporal_median', 'mog2', 'none'],
                        help="Source for the sky/foreground separation. 'temporal_median' and 'none' disable the mask.")

    # --- MASK PARAMETER ARGUMENTS (Unchanged) ---
    parser.add_argument('--motion-threshold', type=float, default=0.1,
                        help="Optical flow motion threshold for sky/foreground mask. (Used with --mask-source optical_flow)")
    parser.add_argument('--mog2-history', type=int, default=50,
                        help="Number of frames used to build the background model for MOG2. (Used with --mask-source mog2)")
                        
    # --- Raw Processing Arguments (Unchanged) ---
    parser.add_argument('--half-size', action='store_true',
                        help="Process raw files at half resolution for speed.")
    parser.add_argument('--no-auto-bright', action='store_true',
                        help="Disable rawpy's automatic brightness adjustment.")
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

    # --- 4. Stack Light Frames (Unchanged) ---
    print(f"Starting light frame stacking using '{args.align_method}' alignment...")
    # 'mask_or_ref' now holds EITHER the uint8 mask OR the float32 ref image
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
        verbose=True
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
        # Check if it's the reference image (3-channel float) or a mask (2-channel uint8)
        if mask_or_ref.ndim == 3 and mask_or_ref.dtype == np.float32:
            # It's the temporal median reference image. Save it as a 16-bit image.
            print(f"Saving temporal median reference image to {args.save_mask}...")
            stretched_ref = contrast_stretch(mask_or_ref)
            output_ref_16bit = float_to_int16(stretched_ref)
            if output_ref_16bit.shape[2] == 3:
                output_ref_16bit = output_ref_16bit[...,::-1] # BGR conversion
            cv.imwrite(args.save_mask, output_ref_16bit)
        else:
            # It's a regular mask (uint8)
            print(f"Saving alignment mask to {args.save_mask}...")
            cv.imwrite(args.save_mask, mask_or_ref)
            
    elif args.save_mask and mask_or_ref is None and args.mask_source not in ['none']:
        print(f"Note: --save-mask was specified, but the '{args.mask_source}' asset generation failed or was disabled.")
        
    print("✅ Image Stacker Done.")


if __name__ == '__main__':
    import multiprocessing
    # Set start method for better compatibility across OS (especially Windows)
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass 
        
    main()