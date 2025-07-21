# 🐶🐱 Cats vs Dogs Image Classification using CNN

This project demonstrates how to build a **Convolutional Neural Network (CNN)** using **TensorFlow/Keras** to classify images of cats and dogs. It also integrates a **Gradio UI** for interactive prediction and includes instructions to deploy the app on **Hugging Face Spaces**.

## 📁 Files Included

- `Image_Classification_of_Cats_and_Dogs_using_CNN_CODE.ipynb` – Main Jupyter notebook containing the full code and explanation.
- `app.py` *(optional)* – Python script version for Gradio deployment (if converted from notebook).
- `requirements.txt` – List of required libraries.

## 🧠 Project Overview

This notebook walks through the process of building an image classification model to distinguish between images of cats and dogs using a custom CNN architecture. It uses TensorFlow’s `ImageDataGenerator` for preprocessing and **Gradio** to deploy a simple web interface for real-time inference.

### Workflow Summary

- Data loading and preprocessing  
- CNN model building using Keras  
- Model training and evaluation  
- Deployment using Gradio  
- Optional deployment on Hugging Face Spaces

## 🛠️ Technologies Used

- Python 3.x  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Gradio (for deployment)  
- Hugging Face Spaces (for hosting)

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/mincater/cats-vs-dogs-cnn.git
cd cats-vs-dogs-cnn
```

### 2. Install Dependencies

Use the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install tensorflow keras numpy matplotlib gradio
```

### 3. Prepare the Dataset

Ensure your dataset is organized like this:

```
dataset/
├── training_set/
│   ├── cats/
│   └── dogs/
├── test_set/
│   ├── cats/
│   └── dogs/
```

Adjust paths in the notebook if necessary.

### 4. Run the Notebook

```bash
jupyter notebook
```

Open and run `Image_Classification_of_Cats_and_Dogs_using_CNN_CODE.ipynb`.

### 5. Run the Gradio App (Optional)

If your model is saved (e.g., as `model.h5`), you can build a simple Gradio interface:

```python
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("model.h5")

def classify_image(img):
    img = img.resize((64, 64)).convert("RGB")
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model.predict(img_array)[0][0]
    return "Dog" if prediction >= 0.5 else "Cat"

interface = gr.Interface(fn=classify_image, inputs="image", outputs="text")
interface.launch()
```

Save this in `app.py` and run:

```bash
python app.py
```

### 6. 🚀 Deploy to Hugging Face Spaces

To deploy on [Hugging Face Spaces](https://huggingface.co/spaces):

1. Create a new Space (choose **Gradio** SDK).
2. Upload:
   - `app.py`
   - `model.h5`
   - `requirements.txt`
3. Ensure `requirements.txt` includes:

```txt
gradio
tensorflow
numpy
Pillow
```

4. Hugging Face will automatically build and deploy your app.

## 📊 Output

- Accuracy/loss visualization  
- Real-time predictions via web interface  
- Interactive demo hosted online (optional)

## 📌 Notes

- This model is for educational/demo purposes. For better accuracy, consider transfer learning with pre-trained models.
- Gradio makes it easy to test and share ML models without needing a web development background.

