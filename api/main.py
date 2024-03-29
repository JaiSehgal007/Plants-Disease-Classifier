from fastapi import FastAPI, File , UploadFile
import tensorflow as tf
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

MODEL=tf.keras.models.load_model("../models/1")

CLASS_NAMES=['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

@app.get("/ping")
async def ping():
    return "The Server is Running"

def read_file_as_image(data)-> np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image
    
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch=np.expand_dims(image,0)
    prediction=MODEL.predict(img_batch)
    predicted_class=CLASS_NAMES[np.argmax(prediction[0])]
    confidence=np.round(np.max(prediction[0])*100,2)
    return {
        'class':predicted_class,
        'confidence':float(confidence)
    }
    

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
