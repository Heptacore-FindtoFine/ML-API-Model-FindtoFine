from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage # type: ignore
import tensorflow as tf  # type: ignore
import os

app = FastAPI()

# Inisialisasi Google Cloud Storage client
storage_client = storage.Client()

# Fungsi untuk mengunduh model dari Google Cloud Storage
def download_model():
    model_path = "model.h5"
    bucket_name = "findtofine"
    blob_name = "resnet50FindtoFine.h5"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(model_path)
    return model_path

# Download model jika belum ada
if not os.path.exists("model.h5"):
    download_model()

# Load model
model = tf.keras.models.load_model("model.h5")

# Endpoint untuk mengecek apakah server berjalan
@app.get("/")
def read_root():
    return {"message": "Server is running"}

# Contoh request body
class PredictionRequest(BaseModel):
    data: list

# Endpoint untuk memprediksi data
@app.post("/predict/")
def predict(request: PredictionRequest):
    try:
        # Konversi data input ke dalam tensor
        input_data = tf.constant(request.data, dtype=tf.float32)
        # Lakukan prediksi menggunakan model
        predictions = model(input_data).numpy().tolist()
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Menjalankan server (gunakan perintah ini hanya untuk local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)