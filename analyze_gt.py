import cv2
import numpy as np
import glob
import os

inputs_dir = r"c:\Users\ASUS\Documents\Studies&Exams\internship\temp\inputs"
gt_dir = r"c:\Users\ASUS\Documents\Studies&Exams\internship\temp\ground truth"

def get_gt_point(im_name):
    base = os.path.splitext(im_name)[0]
    im_path = os.path.join(inputs_dir, base + ".jpeg")
    gt_path = os.path.join(gt_dir, base + ".jpg")
    
    img = cv2.imread(im_path)
    gt = cv2.imread(gt_path)
    
    if img is None or gt is None:
        print(f"Could not read {im_name}")
        return None
        
    diff = cv2.absdiff(img, gt)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Threshold the difference to find the mark
    _, thresh = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the largest contour
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY)
        else:
            return (c[0][0][0], c[0][0][1])
    return None

if __name__ == "__main__":
    for f in os.listdir(gt_dir):
        if f.endswith(".jpg"):
            pt = get_gt_point(f)
            print(f"{f}: {pt}")
