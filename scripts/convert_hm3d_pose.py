import os
import glob
import numpy as np
from natsort import natsorted

def convert_pose_folder_to_txt(scene_dir):
    pose_dir = os.path.join(scene_dir, "pose")
    output_path = os.path.join(scene_dir, "pose.txt")
    
    if not os.path.exists(pose_dir):
        print(f"Pose directory not found: {pose_dir}")
        return

    pose_files = natsorted(glob.glob(os.path.join(pose_dir, "*.txt")))
    
    print(f"Found {len(pose_files)} pose files in {pose_dir}")
    
    with open(output_path, "w") as out_f:
        for i, pose_file in enumerate(pose_files):
            # Create a fake timestamp, e.g., frame index * 0.1s
            timestamp = float(i) * 0.1
            
            with open(pose_file, "r") as f:
                content = f.read().strip()
                # Assuming content is "r00 r01 ... r33" (16 floats)
                # Just copy it
                # Ensure it's single line space separated
                values = content.split()
                if len(values) != 16:
                    print(f"Warning: {pose_file} does not have 16 values")
                    continue
                
                line = f"{timestamp} {' '.join(values)}\n"
                out_f.write(line)
                
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    scene_path = "/path/to/OASIS/dataset/HM3D_hovsg/00824-Dd4bFSTQ8gi"
    convert_pose_folder_to_txt(scene_path)

