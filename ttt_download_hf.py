import os
import time
import subprocess

datasets = [
    "AterMors/wikiart_recaption",
    "lcolonn/patfig",
    "visual-layer/oxford-iiit-pet-vl-enriched",
    "alfredplpl/artbench-pd-256x256",
    "bigdata-pw/TheSimpsons",
    "vera365/lexica_dataset",
    "rezashkv/styles",
    "hahminlew/kream-product-blip-captions",
    "zoheb/sketch-scene"
]

local_dir = "/home/ma-user/work/dataset/all/hf-datasets"


def download_dataset(dataset):
    cmd = [
        "huggingface-cli", "download",
        "--repo-type", "dataset",
        "--resume-download",
        dataset,
        "--local-dir", local_dir
    ]
    subprocess.run(cmd, check=True)


while True:
    for dataset in datasets:
        try:
            print(f"Starting download for {dataset}")
            download_dataset(dataset)
            print(f"Completed download for {dataset}")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {dataset}: {e}")

    print("All datasets downloaded. Sleeping for 10 seconds...")
    time.sleep(10)
