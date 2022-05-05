import torch
import torchvision.models as models
import onnxruntime as onnxrt
from PIL import Image
from torchvision import transforms
import numpy as np
from onnx2keras import onnx_to_keras
import onnx
from utils import decode_predictions
import tensorflow
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = models.resnet50(pretrained=True)
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)
input_names = ["actual_input"]
output_names = ["output"]

torch.onnx.export(model,
                  dummy_input,
                  "./saved_model/torch2keras_resnet50_onnx.onnx",
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names,
                  export_params=True,
                  )

img_cat = Image.open("./img/cat.jpeg").convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

img_cat_preprocessed = preprocess(img_cat)
batch_img_cat_tensor = torch.unsqueeze(img_cat_preprocessed, 0)
pt_output_tensor = model(batch_img_cat_tensor)
pt_output_np = pt_output_tensor.detach().cpu().numpy()
pt_label = decode_predictions(pt_output_np)

print('----------- pytorch results -----------')
for prediction_id in range(len(pt_label[0])):
    print(pt_label[0][prediction_id])

img_np = batch_img_cat_tensor.detach().cpu().numpy()
onnx_session = onnxrt.InferenceSession("./saved_model/torch2keras_resnet50_onnx.onnx")

onnx_inputs = {onnx_session.get_inputs()[0].name: img_np}
onnx_output = onnx_session.run(None, onnx_inputs)
onnx_label = decode_predictions(onnx_output[0])

print('----------- onnx results -----------')
for prediction_id in range(len(onnx_label[0])):
    print(onnx_label[0][prediction_id])

onnx_model = onnx.load('./model/resnet50_onnx.onnx')
k_model = onnx_to_keras(onnx_model, ['actual_input'], input_shapes=[(3, 224, 224)], verbose=True, change_ordering=True)
tensorflow.keras.models.save_model(k_model,'./saved_model/torch2keras_resnet50_keras.h5', overwrite=True, include_optimizer=True)
k_model = tensorflow.keras.models.load_model('./saved_model/torch2keras_resnet50_keras.h5')

keras_out = k_model.predict(np.transpose(img_np, (0, 2, 3, 1)))
keras_label = decode_predictions(keras_out)

print('----------- keras results -----------')
for prediction_id in range(len(keras_label[0])):
    print(keras_label[0][prediction_id])
