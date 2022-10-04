import numpy as np
from fastapi import FastAPI,UploadFile,File
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf


MODELS = tf.keras.models.load_model("E:/CodeBasics/potato-disease-prediction/saved_models/1")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]



app = FastAPI()

@app.get('/ping')
async def ping():
    return "hello,i am a live"


def read_files_as_image(data) ->np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post('/predict')
async def predict(
        file:UploadFile = File(...)
):
    image = read_files_as_image(await file.read())
    img_batch = np.expand_dims(image,0) #0 means axis=0 ,means row
    predictions = MODELS.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions[0])
    print(predicted_class,confidence)

    return {
        "class" :predicted_class,
        "confidence" :float(confidence)
    }




if __name__ == '__main__':
    uvicorn.run(app,host='localhost',port=8000)