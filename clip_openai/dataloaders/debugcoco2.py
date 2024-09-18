import json

# 读取 JSON 文件
with open('/Users/lwz/torch_ds/coco/annotations/dataset_coco.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 打印文件内容
print(json.dumps(data, indent=4, ensure_ascii=False))
