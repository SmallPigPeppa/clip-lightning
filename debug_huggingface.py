from datasets import load_dataset
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HF_HOME'] = '/ppio_net0/huggingface'
# ds = load_dataset("pixparse/cc3m-wds")


# ds = load_dataset("UCSC-VLAA/Recap-COCO-30K")
# ds = load_dataset("Tverous/flicker30k")
# ds = load_dataset("phiyodr/coco2017")
# ds = load_dataset("OpenFace-CQUPT/FaceCaption-15M")
# ds = load_dataset("jmhessel/newyorker_caption_contest", "explanation")
# ds = load_dataset("TheFusion21/PokemonCards")
# ds = load_dataset("jinaai/fashion-captions-de")
# ds = load_dataset("lcolonn/patfig")
# ds = load_dataset("AterMors/wikiart_recaption")
# ds = load_dataset("Norod78/microsoft-fluentui-emoji-512-whitebg")
# ds = load_dataset("MohamedRashad/midjourney-detailed-prompts")
# ds = load_dataset("m1guelpf/nouns")


# from transformers import AutoModel, AutoTokenizer
# import ssl
# import os
#
# # Disable SSL verification globally
# ssl._create_default_https_context = ssl._create_unverified_context
# # Optionally, set environment variables to disable SSL certificate verification
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # To suppress warnings
# ds = load_dataset("Norod78/microsoft-fluentui-emoji-512-whitebg")

# ds = load_dataset("Norod78/microsoft-fluentui-emoji-512-whitebg",split="train")
# print(ds[0])
# ds = load_dataset("AterMors/wikiart_recaption",split="train")
# print(ds[0])
# ds = load_dataset("OpenFace-CQUPT/FaceCaption-15M",split="train")
# print(ds[0])
# ds = load_dataset("jmhessel/newyorker_caption_contest",split="train")
# print(ds[0])



from clip_openai.dataloaders.base_hf import DATASET_MAPPINGS
for dataset_name, dataset_info in DATASET_MAPPINGS.items():
    print(f"Downloading dataset: {dataset_name}")
    try:
        ds = load_dataset(dataset_info['hf_name'])
        print(f"Successfully downloaded {dataset_name}")
    except Exception as e:
        print(f"Failed to download {dataset_name}: {e}")

print("All downloads attempted.")