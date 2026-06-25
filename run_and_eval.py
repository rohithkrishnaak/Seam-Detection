import os
import cv2
import numpy as np
import sys
import math

# Add the directory containing seam_detector2.py to path so we can import it
sys.path.append(r"c:\Users\ASUS\Documents\Studies&Exams\internship\temp")
import seam_detector2

inputs_dir = r"c:\Users\ASUS\Documents\Studies&Exams\internship\temp\inputs"
gt_dir = r"c:\Users\ASUS\Documents\Studies&Exams\internship\temp\ground truth"
output_dir = r"c:\Users\ASUS\Documents\Studies&Exams\internship\temp\output3"

os.makedirs(output_dir, exist_ok=True)

def get_gt_point(im_name):
    base = os.path.splitext(im_name)[0]
    im_path = os.path.join(inputs_dir, base + ".jpeg")
    gt_path = os.path.join(gt_dir, base + ".jpg")
    
    img = cv2.imread(im_path)
    gt = cv2.imread(gt_path)
    
    if img is None or gt is None:
        return None
        
    diff = cv2.absdiff(img, gt)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY)
        else:
            return (c[0][0][0], c[0][0][1])
    return None

def main():
    results = []
    
    # Process each image in inputs
    for f in os.listdir(inputs_dir):
        if not f.lower().endswith(".jpeg"):
            continue
            
        base = os.path.splitext(f)[0]
        im_path = os.path.join(inputs_dir, f)
        
        # Ground truth
        gt_pt = get_gt_point(base + ".jpg")
        
        # Predict
        img = cv2.imread(im_path)
        pred = seam_detector2.detect_seam(img)
        
        pred_pt = None
        if pred is not None:
            pred_pt = (pred["seam_x"], pred["y_bend"])
            
            # Save output image
            annotated = seam_detector2.draw_seam(img, pred)
            out_path = os.path.join(output_dir, base + "_seam.jpg")
            cv2.imwrite(out_path, annotated)
        
        if gt_pt is None:
            print(f"{base}: Ground truth not found")
            continue
            
        if pred_pt is None:
            print(f"{base}: Seam not detected")
            continue
            
        dist = math.sqrt((gt_pt[0] - pred_pt[0])**2 + (gt_pt[1] - pred_pt[1])**2)
        print(f"{base}: GT={gt_pt}, Pred=({pred_pt[0]:.1f}, {pred_pt[1]:.1f}), Error={dist:.2f} px")
        results.append(dist)
        
    if results:
        avg_err = sum(results) / len(results)
        print(f"\nAverage Error: {avg_err:.2f} pixels over {len(results)} images.")
        
if __name__ == "__main__":
    main()
