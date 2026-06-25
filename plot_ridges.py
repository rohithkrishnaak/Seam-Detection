import os, cv2, numpy as np, sys
sys.path.append(r"c:\Users\ASUS\Documents\Studies&Exams\internship\temp")
import seam_detector2

inputs_dir = r"c:\Users\ASUS\Documents\Studies&Exams\internship\temp\inputs"
out_dir = r"c:\Users\ASUS\Documents\Studies&Exams\internship\temp\debug_out"
os.makedirs(out_dir, exist_ok=True)

for name in ["im10", "im11", "im22"]:
    im_path = os.path.join(inputs_dir, name + ".jpeg")
    img = cv2.imread(im_path)
    
    subx, suby = seam_detector2._multiscale_steger_ridge_points(
        img, sigma=2.0, lambda_thresh=-2.0, angle_filter_deg=45.0)
    
    # Draw points on image
    out = img.copy()
    for x, y in zip(subx, suby):
        cv2.circle(out, (int(x), int(y)), 1, (0, 255, 0), -1)
        
    cv2.imwrite(os.path.join(out_dir, name + "_ridges.jpg"), out)
    print(f"Saved {name} ridges.")
