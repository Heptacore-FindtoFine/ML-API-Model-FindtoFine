import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
from dotenv import load_dotenv

# Muat variabel lingkungan dari file .env
load_dotenv()

# Inisialisasi Google Cloud Storage client dengan service account
os.environ.setdefault("GCLOUD_PROJECT", os.getenv("GCLOUD_PROJECT"))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

app = FastAPI()

# Inisialisasi Google Cloud Storage client
storage_client = storage.Client()

# Fungsi untuk mengunduh model dari Google Cloud Storage
def download_model():
    model_path = "model.h5"
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    blob_name = os.getenv("GCS_MODEL_PATH")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    print("Mengunduh model dari Google Cloud Storage...")
    blob.download_to_filename(model_path)
    
    print("Model berhasil diunduh.")
    return model_path

# Download model jika belum ada
if not os.path.exists("model.h5"):
    download_model()

# Load model
model = tf.keras.models.load_model("model.h5", compile=False)

# Daftar kelas untuk prediksi
class_names = [
    'air mineral', 'jaket gunung', 'jas hujan', 'kompor portable',
    'lampu senter', 'sarung tangan gunung', 'sepatu gunung',
    'sleeping bag', 'tas gunung', 'tenda', 'tikar', 'topi gunung'
]

class ImageRequest(BaseModel):
    image_url: str

@app.post("/predict")
async def predict(image_request: ImageRequest):
    try:
        response = requests.get(image_request.image_url)
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download image")
        
        try:
            image = Image.open(BytesIO(response.content))
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Cannot identify image file")
        
        image = image.resize((224, 224))
        image_array = tf.keras.utils.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions[0])
        probability = predictions[0][predicted_class]

        if probability >= 0.8:
            class_name = class_names[predicted_class]
        else:
            class_name = 'barang tidak diketahui'
        
        return {"predicted_class": class_name, "probability": float(probability)}

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error downloading image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
