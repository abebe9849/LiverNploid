import glob
import numpy as np
import pandas as pd
import tqdm,json
from pathlib import Path
from cellseg_models_pytorch.utils import FileHandler
from skimage.measure import label, regionprops
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

valid_img_dir = "/data/RSNA/Nploid_test/forHEIP?_ds2"
save_dir = "/data/RSNA/Nploid_test/save_dir4"

def analyze_nuclei(mask):
    labeled_mask = label(mask)
    properties = []
    
    for region in regionprops(labeled_mask):
        area = region.area
        perimeter = region.perimeter
        solidity = region.solidity
        eccentricity = region.eccentricity
        major_axis = region.major_axis_length
        minor_axis = region.minor_axis_length
        aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        properties.append({
            "area": area,
            "perimeter": perimeter,
            "circularity": circularity,
            "solidity": solidity,
            "eccentricity": eccentricity,
            "major_axis": major_axis,
            "minor_axis": minor_axis,
            "aspect_ratio": aspect_ratio
        })
    
    return properties

def process_mask(mask_path):
    masks1 = FileHandler.read_mat(mask_path, return_all=True)
    mask = masks1["inst_map"] * (masks1["type_map"] == 1)
    return analyze_nuclei(mask)

import os
def process_wsi(wsi_id):
    if os.path.exists(f"/data/RSNA/Nploid_test/jsonOut/neoplastic/{wsi_id}_properties.json"):
        return None
    masks_list = glob.glob(f"{save_dir}/{wsi_id}*")
    
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(executor.map(process_mask, masks_list))

    
    all_properties = [prop for sublist in results for prop in sublist]
    #save all_properties to json

    if not all_properties:
        return None
    json_serializable_properties = []
    for prop in all_properties:
        serializable_prop = {}
        for key, value in prop.items():
            # Convert NumPy types to native Python types
            if isinstance(value, (np.integer, np.int64)):
                serializable_prop[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                serializable_prop[key] = float(value)
            else:
                serializable_prop[key] = value
        json_serializable_properties.append(serializable_prop)
    
    def compute_stats(key):
        values = [prop[key] for prop in all_properties]
        return np.median(values), np.std(values)
    
        #np.save(f"/data/RSNA/Nploid_test/properties/{wsi_id}_{k}.npy", )
    with open(f"/data/RSNA/Nploid_test/jsonOut/neoplastic/{wsi_id}_properties.json", "w") as f:
        json.dump(json_serializable_properties, f, indent=2)
    return {
        "WSI_ID": wsi_id,
        **{f"{key}_med": compute_stats(key)[0] for key in all_properties[0].keys()},
        **{f"{key}_std": compute_stats(key)[1] for key in all_properties[0].keys()}
    }

if __name__ == "__main__":
    wsi_files = glob.glob(f"{save_dir}/*")
    all_wsi = list(set([Path(w).stem.replace("T_2","T").split("T_画像")[0] for w in wsi_files]))
    IDX = 3 # 10まで
    print(len(all_wsi))#MT_AI_061_T画像はやり直し
    TT = 10
    #all_wsi = all_wsi[:3]
    #all_wsi = all_wsi[IDX*TT:IDX*TT+TT]
    #exit()
    results = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        future_to_wsi = {executor.submit(process_wsi, wsi): wsi for wsi in all_wsi}
        for future in tqdm.tqdm(as_completed(future_to_wsi), total=len(all_wsi)):
            result = future.result()
            if result:
                results.append(result)
    
    res_df = pd.DataFrame(results)

    #res_df.to_csv(f"/home/abebe9849/Nploid/HEIP/properties_{IDX}.csv", index=False)
