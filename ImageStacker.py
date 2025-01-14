# 🚀 ImageStacker.py: A Python Engine to Stack Images for Astrophotography
# Author: Timothy Do

# Importing Libraries
import os
import numpy as np
import cv2 as cv
import rawpy
from tqdm import tqdm
import torch
import kornia as K
import gc

# Normalize an Image (float) from 0-1
def normalize(img):
    return (img - img.min())/(img.max()-img.min())

# Contrast Stretches an Image (float)
def contrast_stretch(img,min=0,max=1):
    return np.clip(normalize(img) * (max-min) + min,a_min=0,a_max=1)

# Convert a uint8/uint16 image to a float image
def int_to_float(img):
    return normalize(img.astype(np.float))

# Convert a float image to a uint8 image
def float_to_int(img):
    return ((2**8-1)*normalize(img)).astype(np.uint8)

# Convert a float image to a uint16 image
def float_to_int16(img):
    return ((2**16-1)*normalize(img)).astype(np.uint16)

# Converts Raw to Numpy Array
def raw_to_numpy(rawPath,uint=False,gamma=(1,1),half_size=True,use_camera_wb=False,no_auto_bright=False):
    with rawpy.imread(rawPath) as raw:
        if(uint):
            out = raw.postprocess() # uint8
        else:
            out = raw.postprocess(gamma=gamma,use_camera_wb=use_camera_wb,use_auto_wb=True,no_auto_bright=no_auto_bright,half_size=half_size,output_bps=16,demosaic_algorithm=rawpy.DemosaicAlgorithm.VNG,output_color=rawpy.ColorSpace.sRGB)
            out = out.astype(np.float32)/ 2**16
        if(out.shape[2] > out.shape[1]): # Vertical, must rotate
            out = np.rot90(out,k=3)
    return out

# Convert Directory of Raws into a Tensor
def rawDir_to_tensor(stackPath,gamma=(1,1),use_camera_wb=False,half_size=True,no_auto_bright=False,exclude=[],verbose=False):
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

# Registers a Float Img (img2) with Respect to a Reference (img1)
def register_image(img1,img2,numFeatures=5000,match=0.9,refill=True):
    # Convert to uint8 grayscale.
    gray1 = cv.cvtColor(float_to_int(img1), cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(float_to_int(img2), cv.COLOR_BGR2GRAY)
    height, width = gray1.shape
    
    # Create ORB detector
    orb_detector = cv.ORB_create(numFeatures)
    
    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(gray1, None)
    kp2, d2 = orb_detector.detectAndCompute(gray2, None)
    
    # Match features between the two images.
    # We create a Brute Force matcher with 
    # Hamming distance as measurement mode.
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
    
    # Match the two sets of descriptors.
    matches = list(matcher.match(d1, d2))

    
    # Sort matches on the basis of their Hamming distance.
    matches.sort(key = lambda x: x.distance)
    
    # Take the top matches forward.
    matches = matches[:int(len(matches)*match)]
    no_of_matches = len(matches)
    
    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
    
    for i in range(len(matches)):
      p1[i, :] = kp1[matches[i].queryIdx].pt
      p2[i, :] = kp2[matches[i].trainIdx].pt
    
    # Find the homography matrix.
    homography, mask = cv.findHomography(p1, p2, cv.RANSAC)
    
    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv.warpPerspective(img1,
                        homography, (width, height))

    if(refill): 
        transformed_img = np.where(transformed_img != 0, transformed_img, img2) 
    
    return transformed_img

# Stacks Images with Automatic Alignment
def stack_images(imgStack,method='mean',align=True,refIndex=0.5,orbFeatures=5000,orbMatch=0.9,refill=True):
    if align:
        refIndex = len(imgStack) // 2 if refIndex == 0.5 else refIndex
        newImgs = []
        for i in tqdm(range(imgStack.shape[0])): 
            newImgs.append(normalize(imgStack[i]) if i == refIndex else register_image(imgStack[i],imgStack[refIndex],numFeatures=orbFeatures,match=orbMatch,refill=refill))
        imgStack = np.stack(newImgs)
    stackedImg = np.median(imgStack,axis=0) if method == 'med' else np.mean(imgStack,axis=0)
    return stackedImg

# Stack Images with Automatic Alignment (Memory Saver with Mean Method)
def stack_images_opt(stackFiles,gamma=(1,1),use_camera_wb=False,half_size=True,no_auto_bright=False,align=True,refIndex=0.5,orbFeatures=5000,orbMatch=0.9,refill=True,stackedDark=None):
    stackedImg = 0
    rawSettings = (False,gamma,half_size,use_camera_wb,no_auto_bright)
    refIndex = len(stackFiles) // 2 if refIndex == 0.5 else refIndex
    refImg = np.clip(raw_to_numpy(stackFiles[refIndex],*rawSettings) - (stackedDark if stackedDark is not None else 0),a_min=0,a_max=1)
    for i in tqdm(range(len(stackFiles))):
        if i == refIndex:
            stackedImg += refImg/len(stackFiles) 
        else: 
            capImg = np.clip(raw_to_numpy(stackFiles[i],*rawSettings) - (stackedDark if stackedDark is not None else 0),a_min=0,a_max=1)
            stackedImg += (register_image(capImg,refImg,numFeatures=orbFeatures,match=orbMatch,refill=refill) if align else capImg)/len(stackFiles)
        gc.collect()
    return stackedImg

if __name__ == '__main__': 
    dataDir = './data'
    os.makedirs(dataDir,exist_ok=True)
    testDir = './data/orionbelt1'
    stackFiles = sorted([f'{testDir}/{file}' for file in os.listdir(testDir) if file.endswith('.CR2')])
    stackedImg = stack_images_opt(stackFiles,half_size=True,use_camera_wb=True,no_auto_bright=False,align=True,orbFeatures=100000,orbMatch=0.9)
    cv.imwrite(f'{testDir}/stackedImg.tiff',float_to_int16(stackedImg[...,::-1]))

