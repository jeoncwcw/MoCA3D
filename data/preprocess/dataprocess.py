import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import os
    
    
def save_dataset(data:Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
def project_corners(K: np.ndarray, corners_3d: np.ndarray) -> List[Dict[str, float]]:
    projected_corners: List[Dict[str, float]] = []
    for corner in corners_3d:
        x, y, z = corner
        if z <= 0:
            continue
        uvw = K @ np.array([x, y, z], dtype=float)
        u = float(uvw[0] / uvw[2])
        v = float(uvw[1] / uvw[2])
        projected_corners.append({"u": u, "v": v})
    return projected_corners


def build_output(dataset: Dict[str, Any]) -> Dict[str, Any]:
    image_map = {image["id"]: image for image in dataset.get("images", [])}
    result_objects = []
    
    for obj in dataset.get("annotations", []):
        # Get intrinsics
        new_obj = obj.copy()
        image_info = image_map.get(obj["image_id"])
        if not image_info or "K" not in image_info:
            continue
        K = np.asarray(image_info.get("K"), dtype=float)
        
        # Create projected corners and depth list
        depth = []
        corners = obj.get("bbox3D_cam", [])
        for corner in corners:
            depth.append(corner[2])
        new_obj["depth"] = depth
        
        if corners and len(corners) == 8 and K.shape == (3, 3):
            projected = project_corners(K, corners)
            if len(projected) == 8:
                new_obj["projected_corners"] = projected
            else:
                continue
        
        # Build new object
        result_objects.append(new_obj)
    
    output_data = dataset.copy()
    output_data["annotations"] = result_objects
    return output_data

def main() -> None:
    parser = argparse.ArgumentParser(description="Process 3D bounding box dataset.")
    parser.add_argument("--input_dir", type=Path, help="Path to the input JSON dataset")
    parser.add_argument("--output_dir", type=Path, help="Path to save the processed dataset")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = [f for f in os.listdir(args.input_dir) if not f.endswith("stats.json")]
    for file_name in files:
        print(f"Processing {file_name}...")
        input_path = args.input_dir / file_name
        output_path = args.output_dir / file_name
        with input_path.open("r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        processed_dataset = build_output(dataset)
        save_dataset(processed_dataset, output_path)
        print(f"Saved processed dataset to {output_path}")


if __name__ == "__main__":
    main()