from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

MODEL = tf.keras.models.load_model("../saved_model/1.keras")

CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

@app.get('/ping')
async def ping():
    return 'Hello i am alive'

def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)))
    print(image.shape)
    return image



@app.post('/predict')
async def predict(
    file: UploadFile = File(...)
):
    #print("hello")
    image = read_file_as_image(await file.read())
    #print(image)
    img_batch = np.expand_dims(image, 0)
    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }