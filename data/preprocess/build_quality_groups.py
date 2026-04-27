import argparse
import json
from typing import Dict, List
from pathlib import Path

def assign_group(
    corners: List[Dict], width: float, height: float
    ) -> str:
    inside = 0
    for c in corners:
        u, v = float(c["u"]), float(c["v"])
        if 0.0 <= u < width and 0.0 <= v < height:
            inside += 1
    if inside == 8:
        return "Good"
    if 4 <= inside <= 7:
        return "Moderate"
    return "Poor"


def split_from_path(annotation_path: Path) -> str:
    name = Path(annotation_path).name
    if name.endswith("_train.json"):
        return "train"
    if name.endswith("_val.json"):
        return "val"
    if name.endswith("_test.json"):
        return "test"

  
def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--root_dir", type=Path, required=False, default="./datasets/tmp",
    )
    args = argparser.parse_args()
    root_dir = args.root_dir

    for ann_path in sorted(root_dir.glob("*.json")):
        # Load annotations
        with open(ann_path, "r") as f:
            annotations = json.load(f)
            image_info_map = {img["id"]: img for img in annotations["images"]}
            objects = annotations["annotations"]

        # Build quality groups
        quality_group = {"Good": [], "Moderate": [], "Poor": []}

        for obj in objects:
            img = image_info_map[obj["image_id"]]
            obj_corners = obj["projected_corners"]
            group = assign_group(obj_corners, width=img["width"], height=img["height"])
            obj["quality"] = group
            quality_group[group].append(obj["id"])

        # Print Stats and save
        split = split_from_path(ann_path).capitalize()
        print("=" * 10 + f" {split} " + "=" * 10 + f"  ({ann_path.name})")
        print("Good:", len(quality_group["Good"]))
        print("Moderate:", len(quality_group["Moderate"]))
        print("Poor:", len(quality_group["Poor"]))

        with open(ann_path, "w") as f:
            json.dump(annotations, f, indent=4)
        print(f"Saved labeled annotations to {ann_path}")
        
        
if __name__ == "__main__":
    main()
