import os
import kagglehub
os.environ["KAGGLEHUB_CACHE"] = "/mnt/hdfs/byte_content_security/user/liuwenzhuo/ckpt/kagglehub_cache"
# path = kagglehub.dataset_download("hsankesara/flickr-image-dataset")
# print("Path to dataset files:", path)
# path = kagglehub.dataset_download("nikhil7280/coco-image-caption")
# print("Path to dataset files:", path)


path = kagglehub.dataset_download("lyte69/gqa-images")
print("Path to dataset files:", path)