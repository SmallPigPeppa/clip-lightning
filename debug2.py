import timm
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['SSL_CERT_DIR'] = ''
model_name = 'resnet50'
pretrained = True
model = timm.create_model(
    model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
)
print(model)

model2 = timm.create_model(
    model_name, pretrained=pretrained
)
print(model2)

model_name = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'
pretrained = True
model3 = timm.create_model(
    model_name, pretrained=pretrained
)
print(model3)



model_name = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'
pretrained = True
model4 = timm.create_model(
    model_name, pretrained=pretrained,num_classes=0, global_pool="avg"
)
print(model4)
# print(model4.hparams)