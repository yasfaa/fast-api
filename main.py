import os
from requests import request
import uvicorn
import traceback
import tensorflow as tf
import numpy as np
import io
from PIL import Image
from pydantic import BaseModel
from urllib.request import urlopen
from fastapi import FastAPI, Response, UploadFile, HTTPException, Depends, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from tensorflow.keras.utils import load_img, img_to_array

app = FastAPI()
security = HTTPBasic()

# # IP whitelist
# allowed_ips = ["192.168.0.1", "10.0.0.1"]


categories = {
    0: "Apel",
    1: "Ayam Goreng",
    2: "Mie Bakso",
    3: "Bakwan",
    4: "Batagor",
    5: "Beberuk",
    6: "Bubur",
    7: "Burger",
    8: "Cakwe",
    9: "Capcay",
    10: "Crepes",
    11: "Donat",
    12: "Es Krim",
    13: "Gudeg",
    14: "Gulai Ikan",
    15: "Ikan Goreng",
    16: "Jeruk",
    17: "Kebab",
    18: "Kentang Goreng",
    19: "Kerak Telor",
    20: "Nasi Kuning",
    21: "Nasi Pecel",
    22: "Papeda",
    23: "Rendang",
    24: "Tahu Sumedang"}


model = tf.keras.models.load_model("klasifikasi_makanan2.h5")


@app.get("/")
async def home():
    return {"health_check": "OK"}


@app.post("/predict_image")
def predict_image(request: Request, url: str, response: Response):
    # client_ip = request.client.host
    # if client_ip not in allowed_ips:
    #     raise HTTPException(status_code=403, detail="IP not whitelisted.")

    try:
        image_response = urlopen(url)
        if image_response.getcode() != 200:
            response.status_code = 400
            return "Failed to retrieve image from URL!"

        image_data = image_response.read()
        image = Image.open(io.BytesIO(image_data))
        # Resize the image to match the expected input shape
        image = image.resize((150, 150))
        image_array = img_to_array(image)
        image_expanded = np.expand_dims(image_array, axis=0)
        image_expanded /= 255.0

        result = model.predict(image_expanded)
        return {"result": categories[np.argmax(result)].lower}

    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return f"Internal Server Error: {str(e)}"


port = os.environ.get("PORT", 8080)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host="0.0.0.0", port=int(port))
