import json

# 读取 JSON 文件
with open('/Users/lwz/torch_ds/coco/annotations/dataset_coco.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 检查 images 中的 sentences 列表长度
for image in data['images']:
    if len(image['sentences']) != 5:
        print(f"Image ID: {image['cocoid']}, Sentences Length: {len(image['sentences'])}")
        for i in image['sentences']:
            print(i['raw'])
