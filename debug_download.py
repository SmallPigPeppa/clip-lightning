import timm

# 列出所有想要加载的模型名称
model_names = [
    'vit_base_patch16_224',
    'vit_base_patch16_224.augreg_in21k_ft_in1k',
    'deit_base_distilled_patch16_224.fb_in1k',
    'vit_base_patch16_224.clip_openai_ft_in1k',
    'pvt_v2_b3.in1k',
    'mobilenetv3_small_050.lamb_in1k',
    'resnet18.a1_in1k',
    'resnet50'
]

# 使用字典来存储每个模型名称对应的模型实例
models = {}

# 循环遍历模型名称列表，为每个模型名称加载模型
for model_name in model_names:
    models[model_name] = timm.create_model(model_name, pretrained=True)
    print(f"Loaded model: {model_name}")

# 现在可以通过 models 字典访问每个加载的模型
# 例如：models['vit_base_patch16_224']
