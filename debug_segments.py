import os, cv2, numpy as np, sys
sys.path.append(r"c:\Users\ASUS\Documents\Studies&Exams\internship\temp")
if 'seam_detector2' in sys.modules:
    del sys.modules['seam_detector2']
import seam_detector2

inputs_dir = r"c:\Users\ASUS\Documents\Studies&Exams\internship\temp\inputs"

for name in ["im10"]:
    im_path = os.path.join(inputs_dir, name + ".jpeg")
    img = cv2.imread(im_path)
    
    # Check both passes
    for filt in [None, 45.0]:
        subx, suby = seam_detector2._multiscale_steger_ridge_points(
            img, sigma=2.0, lambda_thresh=-2.0, angle_filter_deg=filt)
        segments = seam_detector2._group_into_segments(subx, suby, img.shape)
        fits = [seam_detector2._fit_line(s["xs"], s["ys"]) for s in segments]
        
        pair = seam_detector2._best_segment_pair(segments)
        
        tag = f"filter={filt}"
        print(f"\n{name} [{tag}]: {len(segments)} segments")
        for i, s in enumerate(segments):
            m, b = fits[i]
            print(f"  seg {i}: x=[{s['x0']:.0f},{s['x1']:.0f}] ymean={s['ymean']:.1f} slope={m:.4f} pts={len(s['xs'])}")
        
        if pair:
            left, right, gap, y_diff, score = pair
            print(f"  PAIR: left x=[{left['x0']:.0f},{left['x1']:.0f}] right x=[{right['x0']:.0f},{right['x1']:.0f}] gap={gap:.1f} ydiff={y_diff:.1f} score={score:.4f}")
        else:
            print(f"  NO PAIR")
