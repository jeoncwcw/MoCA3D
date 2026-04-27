import json
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor

def sort_corners_2d_spatial(bbx_3d, projected_corners, depths):
    combined = []
    for i in range(len(projected_corners)):
        combined.append({
            "u": float(projected_corners[i]["u"]),
            "v": float(projected_corners[i]["v"]),
            "depth": float(depths[i])
        })
    v_sorted = sorted(combined, key=lambda x: x["v"], reverse=True)
    
    bottom_4 = v_sorted[:4]
    top_4 = v_sorted[4:]
    
    bottom_4_final = sorted(bottom_4, key=lambda x: x["u"])
    top_4_final = sorted(top_4, key=lambda x: x["u"])
    final_list = bottom_4_final + top_4_final
    
    new_corners = [{"u": p["u"], "v": p["v"]} for p in final_list]
    new_depths = [p["depth"] for p in final_list]
    
    return new_corners, new_depths

def process_per_object(obj):
    corners = obj.get("projected_corners", [])
    depths = obj.get("depth", [])
    bbx_3d = obj.get("bbox3D_cam", [])
    
    if len(corners) == 8 and len(depths) == 8:
        new_corners, new_depths = sort_corners_2d_spatial(bbx_3d, corners, depths)
        obj["projected_corners"] = new_corners
        obj["depth"] = new_depths
    return obj

def process_dataset(input_dir):
    input_path = Path(input_dir)

    json_files = [f for f in input_path.glob("*.json")]
    
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        with ProcessPoolExecutor() as executor:
            ann = data.get("annotations", [])
            data["annotations"] = list(executor.map(process_per_object, ann))
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
if __name__ == "__main__":
    argparser = ArgumentParser(description="Reorder projected corners and depths in dataset annotations.")
    argparser.add_argument("--root_dir", type=str, default="./datasets/tmp", help="Directory containing input JSON files")
    args = argparser.parse_args()
    input_dir = args.root_dir
    process_dataset(input_dir)
