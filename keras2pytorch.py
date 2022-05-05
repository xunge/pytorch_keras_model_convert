import onnx
import numpy as np
import tf2onnx.convert
from onnx2pytorch import ConvertModel
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import torch
from utils import decode_predictions
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_path = './img/cat.jpeg'   # make sure the image is in img_path
img_size = 224
img = image.load_img(img_path, target_size=(img_size, img_size))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)  # (1, 224, 224, 3)
x = preprocess_input(x)

model_keras = ResNet50(include_top=True, weights='imagenet')
predictions_keras = model_keras.predict(x)
keras_label = decode_predictions(predictions_keras)
for prediction_id in range(len(keras_label[0])):
    print(keras_label[0][prediction_id])

onnx_model, _ = tf2onnx.convert.from_keras(model_keras)

onnx.save(onnx_model, 'saved_model/keras2torch_resnet50_onnx.onnx')
onnx_model = onnx.load('saved_model/keras2torch_resnet50_onnx.onnx')

pytorch_model = ConvertModel(onnx_model)
torch.save(pytorch_model, 'saved_model/keras2torch_resnet50_torch.pth')
pytorch_model = torch.load('saved_model/keras2torch_resnet50_torch.pth')

x_torch = np.transpose(x, (0, 3, 1, 2))  # (3, 224, 224)
img_tensor = torch.from_numpy(x.copy())
predictions_torch = pytorch_model(img_tensor)
predictions_torch = predictions_torch.detach().cpu().numpy()
torch_label = decode_predictions(predictions_torch)
for prediction_id in range(len(torch_label[0])):
    print(torch_label[0][prediction_id])
