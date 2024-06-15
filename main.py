from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import os
import gcsfs
import io
from PIL import Image

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Nama bucket dan path model di Google Cloud Storage
bucket_name = 'findtofine'
model_path = 'resnet50FindtoFine.h5'
local_model_path = '/tmp/model.h5'

# Fungsi untuk mengunduh model dari Google Cloud Storage
def download_model():
    if not os.path.exists(local_model_path):
        fs = gcsfs.GCSFileSystem(project='heptacore-findtofine')
        with fs.open(f'{bucket_name}/{model_path}', 'rb') as f:
            with open(local_model_path, 'wb') as local_f:
                local_f.write(f.read())
    return local_model_path

# Mengunduh dan memuat model jika belum ada
if not os.path.exists(local_model_path):
    download_model()

model = tf.keras.models.load_model(local_model_path)

# Daftar kelas untuk prediksi
classes_names = ['air mineral', 'jaket gunung', 'jas hujan', 'kompor portable',
                 'lampu senter', 'sarung tangan gunung', 'sepatu gunung',
                 'sleeping bag', 'tas gunung', 'tenda', 'tikar', 'topi gunung']

# Fungsi untuk membaca gambar dari GCS
def read_image_from_gcs(bucket_name, image_path):
    fs = gcsfs.GCSFileSystem(project='heptacore-findtofine')
    with fs.open(f'{bucket_name}/{image_path}', 'rb') as f:
        img = Image.open(io.BytesIO(f.read()))
        return img

# Endpoint untuk mengecek apakah server berjalan
@app.get("/")
def read_root():
    return {"message": "Server is running"}

# Contoh request body untuk data prediksi
class PredictionRequest(BaseModel):
    data: list

# Endpoint untuk memprediksi data (sebagai tensor)
# @app.post("/predict_data/")
# def predict_data(request: PredictionRequest):
#     try:
#         input_data = tf.constant(request.data, dtype=tf.float32)
#         predictions = model(input_data).numpy().tolist()
#         return {"predictions": predictions}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

# Contoh request body untuk gambar prediksi
class ImagePathRequest(BaseModel):
    image_path: str

# Endpoint untuk memprediksi kelas dari gambar dengan path GCS
@app.post("/predict_image/")
def predict_image(request: ImagePathRequest):
    try:
        # Mengambil path dari image_path
        image_path = request.image_path  # Ini akan menjadi "taskImage/download.jpeg" dari permintaan HTTP
        
        img = read_image_from_gcs(bucket_name, image_path)

        if img.mode == 'RGBA':
            img = img.convert('RGB')

        img = img.resize((224, 224))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = tf.expand_dims(x, axis=0)
        x = tf.cast(x, tf.float32) / 255.0

        predictions = model.predict(x)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class_name = classes_names[predicted_class_idx]
        probability = predictions[0][predicted_class_idx]

        if probability >= 0.8:
            result = {"class": predicted_class_name}
        else:
            result = {"class": "barang tidak diketahui"}

        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Menjalankan server (gunakan perintah ini hanya untuk local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)