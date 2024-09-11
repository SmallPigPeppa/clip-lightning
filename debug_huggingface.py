
from datasets import load_dataset
import os
os.environ['CURL_CA_BUNDLE'] = ''
# ds = load_dataset("pixparse/cc3m-wds")



# ds = load_dataset("UCSC-VLAA/Recap-COCO-30K")
# ds = load_dataset("Tverous/flicker30k")
# ds = load_dataset("phiyodr/coco2017")
# ds = load_dataset("jinaai/fashion-captions-de")
# ds = load_dataset("OpenFace-CQUPT/FaceCaption-15M")
# ds = load_dataset("lcolonn/patfig")
# ds = load_dataset("jmhessel/newyorker_caption_contest", "explanation")
# ds = load_dataset("AterMors/wikiart_recaption")
# ds = load_dataset("Norod78/microsoft-fluentui-emoji-512-whitebg")

from transformers import AutoModel, AutoTokenizer
import ssl
import os

# Disable SSL verification globally
ssl._create_default_https_context = ssl._create_unverified_context

# Optionally, set environment variables to disable SSL certificate verification
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # To suppress warnings
ds = load_dataset("Norod78/microsoft-fluentui-emoji-512-whitebg")

# ds = load_dataset("Norod78/microsoft-fluentui-emoji-512-whitebg",split="train")
# print(ds[0])
# ds = load_dataset("AterMors/wikiart_recaption",split="train")
# print(ds[0])
# ds = load_dataset("OpenFace-CQUPT/FaceCaption-15M",split="train")
# print(ds[0])
# ds = load_dataset("jmhessel/newyorker_caption_contest",split="train")
# print(ds[0])