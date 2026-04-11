import gradio as gr
from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image

# torch.hub.download_url_to_file('http://images.cocodataset.org/val2017/000000039769.jpg', 'cats.jpg')

feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

def process_image(image):
    # prepare image for the model
    encoding = feature_extractor(image, return_tensors="pt")
    
    # forward pass
    with torch.no_grad():
       outputs = model(**encoding)
       predicted_depth = outputs.predicted_depth
    
    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(1),
                        size=image.size[::-1],
                        mode="bicubic",
                        align_corners=False,
                 ).squeeze()
    output = prediction.cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype('uint8')
    img = Image.fromarray(formatted)
    return img
    
    # return result
    
title = "Demo: zero-shot depth estimation with DPT"
description = "Demo for Intel's DPT, a Dense Prediction Transformer for state-of-the-art dense prediction tasks such as semantic segmentation and depth estimation."
examples =['./Session-04/cats.jpg']

iface = gr.Interface(fn=process_image, 
                     inputs=gr.Image(type="pil"), 
                     outputs=gr.Image(type="pil", label="predicted depth"),
                     title=title,
                     description=description,
                     examples=examples)
iface.launch(debug=True)