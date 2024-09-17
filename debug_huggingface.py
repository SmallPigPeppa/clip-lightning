from datasets import load_dataset
import os
os.environ['CURL_CA_BUNDLE'] = ''
# os.environ['HF_HOME'] = '/ppio_net0/huggingface'
os.environ['HF_HOME']='/home/ma-user/work/dataset/all/hf-datasets'
# ds = load_dataset("pixparse/cc3m-wds")


# ds = load_dataset("UCSC-VLAA/Recap-COCO-30K")
# ds = load_dataset("Tverous/flicker30k")
# ds = load_dataset("phiyodr/coco2017")
# ds = load_dataset("OpenFace-CQUPT/FaceCaption-15M")
# ds = load_dataset("jmhessel/newyorker_caption_contest", "explanation")
# ds = load_dataset("TheFusion21/PokemonCards")
ds = load_dataset("jinaai/fashion-captions-de")
ds = load_dataset("lcolonn/patfig")
ds = load_dataset("AterMors/wikiart_recaption")
# ds = load_dataset("Norod78/microsoft-fluentui-emoji-512-whitebg")
# ds = load_dataset("MohamedRashad/midjourney-detailed-prompts")
# ds = load_dataset("m1guelpf/nouns")
ds = load_dataset("bigdata-pw/TheSimpsons")


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

#
#
# DATASET_MAPPINGS = {
#     'emoji': {
#         'hf_name': 'Norod78/microsoft-fluentui-emoji-512-whitebg',
#         'keys': {'image': 'image', 'text': 'text'},
#         'splits': {'train': 0.5, 'val': 0.5}
#     },
#     'wikiart': {
#         'hf_name': 'AterMors/wikiart_recaption',
#         'keys': {'image': 'image', 'text': 'text'},
#         'splits': {'train': 0.8, 'val': 0.2}
#     },
#     # 'face': {
#     #     'hf_name': 'OpenFace-CQUPT/FaceCaption-15M',
#     #     'keys': {'image': 'url', 'text': 'caption'},
#     #     'splits': {'train': 0.002, 'val': 0.0005}
#     # },
#     # 'newyorker': {
#     #     'hf_name': 'jmhessel/newyorker_caption_contest',
#     #     'keys': {'image': 'image', 'text': 'image_description'},
#     #     'splits': {'train': 'train', 'val': 'validation'}
#     # },
#     'patfig': {
#         'hf_name': 'lcolonn/patfig',
#         'keys': {'image': 'image', 'text': 'short_description'},
#         'splits': {'train': 'train', 'val': 'test'}
#     },
#     'fashion': {
#         'hf_name': 'jinaai/fashion-captions-de',
#         'keys': {'image': 'image', 'text': 'text'},
#         'splits': {'train': 'train', 'val': 'test'}
#     },
#     'pet': {
#         'hf_name': 'visual-layer/oxford-iiit-pet-vl-enriched',
#         'keys': {'image': 'image', 'text': 'caption_enriched'},
#         'splits': {'train': 'train', 'val': 'test'}
#     },
#     # 'midjourney': {
#     #     'hf_name': 'CortexLM/midjourney-v6',
#     #     'keys': {'image': 'image_url', 'text': 'prompt'},
#     #     'splits': {'train': 0.5, 'val': 0.5}
#     # },
#     'nouns': {
#         'hf_name': 'm1guelpf/nouns',
#         'keys': {'image': 'image', 'text': 'text'},
#         'splits': {'train': 0.2, 'val': 0.8}
#     },
#     # 'pokemon': {
#     #     'hf_name': 'TheFusion21/PokemonCards',
#     #     'keys': {'image': 'image_url', 'text': 'caption'},
#     #     'splits': {'train': 0.5, 'val': 0.5}
#     # },
#     'shahnegar': {
#         'hf_name': 'sadrasabouri/ShahNegar',
#         'keys': {'image': 'image', 'text': 'text'},
#         'splits': {'train': 0.8, 'val': 0.2}
#     },
#     # 'peanuts': {
#     #     'hf_name': 'afmck/peanuts-flan-t5-xl',
#     #     'keys': {'image': 'image', 'text': 'caption'},
#     #     'splits': {'train': 0.8, 'val': 0.2}
#     # },
#     # 'vintage': {
#     #     'hf_name': 'SilentAntagonist/vintage-artworks-60k-captioned',
#     #     'keys': {'image': 'image_url', 'text': 'short_caption'},
#     #     'splits': {'train': 0.5, 'val': 0.5}
#     # },
#     'artbench': {
#         'hf_name': 'alfredplpl/artbench-pd-256x256',
#         'keys': {'image': 'image', 'text': 'caption'},
#         'splits': {'train': 0.8, 'val': 0.2}
#     },
#     'hausavg': {
#         'hf_name': 'HausaNLP/HausaVG',
#         'keys': {'image': 'image', 'text': 'en_text'},
#         'splits': {'train': 'train', 'val': 'validation'}
#     },
#     'simpsons': {
#         'hf_name': 'bigdata-pw/TheSimpsons',
#         'keys': {'image': 'jpg', 'text': 'caption.txt'},
#         'splits': {'train': 'train', 'val': 'validation'}
#     },
#     'lexica': {
#         'hf_name': 'vera365/lexica_dataset',
#         'keys': {'image': 'image', 'text': 'prompt'},
#         'splits': {'train': 'train', 'val': 'test'}
#     },
#     'styles': {
#         'hf_name': 'rezashkv/styles',
#         'keys': {'image': 'image', 'text': 'caption'},
#         'splits': {'train': 0.8, 'val': 0.2}
#     },
#     # 'plans': {
#     #     'hf_name': 'ShazShoaib/SingleFloorPlans',
#     #     'keys': {'image': 'image', 'text': 'text'},
#     #     'splits': {'train': 0.8, 'val': 0.2}
#     # },
#     # 'tomato': {
#     #     'hf_name': 'wellCh4n/tomato-leaf-disease-image',
#     #     'keys': {'image': 'image', 'text': 'text'},
#     #     'splits': {'train': 0.8, 'val': 0.2}
#     # },
#     'kream': {
#         'hf_name': 'hahminlew/kream-product-blip-captions',
#         'keys': {'image': 'image', 'text': 'text'},
#         'splits': {'train': 0.8, 'val': 0.2}
#     },
#     'sketch': {
#         'hf_name': 'zoheb/sketch-scene',
#         'keys': {'image': 'image', 'text': 'text'},
#         'splits': {'train': 0.8, 'val': 0.2}
#     },
# }
#
# for dataset_name, dataset_info in DATASET_MAPPINGS.items():
#     print(f"Downloading dataset: {dataset_name}")
#     try:
#         ds = load_dataset(dataset_info['hf_name'])
#         print(f"Successfully downloaded {dataset_name}")
#     except Exception as e:
#         print(f"Failed to download {dataset_name}: {e}")
#
# print("All downloads attempted.")