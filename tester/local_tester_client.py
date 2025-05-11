import os

import requests

url = "http://0.0.0.0:8000/infer_visualize"
script_dir = os.path.dirname(os.path.abspath(__file__))

test_models = ["1k3d68","2d106det","arcface", "buffalo","genderage","glintr100","scrfd_10g_bnkps"]
# Change the current working directory to the parent directory (one level up)
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)


file_path_1 = os.path.join("resources", "test_wc_2011.jpg")
full_path_1 = os.path.join(parent_dir, file_path_1)
for model in test_models:
    with open(full_path_1, 'rb') as f:
        response = requests.post(
            url,
            files={"file": f},
            data={"model_name": model, "image_format": "jpg"}
        )

    print(f"Response from {model}: {response.status_code}")
    print(response.json())



file_path_2 = os.path.join("resources", "test_jw.jpg")
full_path_2 = os.path.join(parent_dir, file_path_2)
for model in test_models:
    with open(full_path_2, 'rb') as f:
        response = requests.post(
            url,
            files={"file": f},
            data={"model_name": model, "image_format": "jpg"}
        )

    print(f"Response from {model}: {response.status_code}")
    print(response.json())
