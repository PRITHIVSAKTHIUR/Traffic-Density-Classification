![dsfsdef.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/bpucqawIvBlE7i0YCG6ba.png)

# **Traffic-Density-Classification**
> **Traffic-Density-Classification** is an image classification vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for a single-label classification task. It is designed to classify images into **traffic density** categories using the **SiglipForImageClassification** architecture.  


```py
Classification Report:
                precision    recall  f1-score   support

  high-traffic     0.8647    0.8410    0.8527       585
   low-traffic     0.8778    0.9485    0.9118      3803
medium-traffic     0.7785    0.6453    0.7057      1187
    no-traffic     0.8730    0.7292    0.7946       528

      accuracy                         0.8602      6103
     macro avg     0.8485    0.7910    0.8162      6103
  weighted avg     0.8568    0.8602    0.8559      6103
```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/xatFDNCVZo5jGHW8njTmz.png)

The model categorizes images into the following 4 classes:  
- **Class 0:** "high-traffic"  
- **Class 1:** "low-traffic"  
- **Class 2:** "medium-traffic"  
- **Class 3:** "no-traffic"  

# **Run with Transformers🤗**

```python
!pip install -q transformers torch pillow gradio
```

```python
import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Traffic-Density-Classification"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def traffic_density_classification(image):
    """Predicts traffic density category for an image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {
        "0": "high-traffic", "1": "low-traffic", "2": "medium-traffic", "3": "no-traffic"
    }
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=traffic_density_classification,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Traffic Density Classification",
    description="Upload an image to classify it into one of the 4 traffic density categories."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
```

# **Intended Use:**  

The **Traffic-Density-Classification** model is designed for traffic image classification. It helps categorize traffic density levels into predefined categories. Potential use cases include:  

- **Traffic Monitoring:** Classifying images from traffic cameras to assess congestion levels.  
- **Smart City Applications:** Assisting in traffic flow management and congestion reduction strategies.  
- **Automated Traffic Analysis:** Helping transportation authorities analyze and optimize road usage.  
- **AI Research:** Supporting computer vision-based traffic density classification models.
