import open3d as o3d
import numpy as np
import os

def inspect_ply(ply_path):
    print(f"Inspecting {ply_path}")
    if not os.path.exists(ply_path):
        print(f"File not found: {ply_path}")
        return

    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        print("Point cloud is empty.")
        return

    points = np.asarray(pcd.points)
    print(f"Number of points: {len(points)}")
    
    min_bound = pcd.get_min_bound()
    max_bound = pcd.get_max_bound()
    
    print("Bounding Box:")
    print(f"Min: {min_bound}")
    print(f"Max: {max_bound}")
    
    extent = max_bound - min_bound
    print(f"Extent: {extent}")
    
    # Determine up axis (usually the one with different range logic, but for indoor scenes, ground is usually flat)
    # If Y-up, ground is XZ plane, so Y variance might be small if it's just a floor, or large if full building.
    # Usually we look at which dimension corresponds to 'height'.
    
    # Let's try to capture a screenshot (offscreen)
    # This might require a display, so we'll skip if it fails.
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        
        output_image = "scene_gt_preview.png"
        vis.capture_screen_image(output_image)
        vis.destroy_window()
        print(f"Saved preview to {output_image}")
    except Exception as e:
        print(f"Could not save preview image: {e}")

if __name__ == "__main__":
    ply_path = "dataset/HM3D_hovsg/00824-Dd4bFSTQ8gi/scene_rgb.ply"
    inspect_ply(ply_path)

