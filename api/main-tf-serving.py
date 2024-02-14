from fastapi import FastAPI, File , UploadFile
import tensorflow as tf
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests

app = FastAPI()

# endpoint="http://localhost:8501/v1/models/plants_model/version/1:predict" -> for multi model structure
endpoint="http://localhost:8501/v1/models/plants_model:predict"

# the name plant_model of the model is the one we specified in the models.config file

# to start tf serving use the command below
# docker run -t --rm -p 8501:8501 -v D:\ML_Projects\Plants-Disease-Classifier:/Plants-Disease-Classifier tensorflow/serving --rest_api_port=8501 --model_config_file=/Plants-Disease-Classifier/models.config


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
    print()
    img_batch=np.expand_dims(image,0)

    json_data = {
        "instances": img_batch.tolist()
    }
    response = requests.post(endpoint,json=json_data)
    print(response.json())
    prediction= np.array(response.json()["predictions"][0])

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }
    

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
