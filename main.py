# Mengimpor modul yang diperlukan
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from google.cloud import storage # type : ignore
import tensorflow as tf # type : ignore
import numpy as np
import os
import gcsfs  # type : ignore
from google.colab import auth # type : ignore

# Autentikasi Google Cloud
auth.authenticate_user()

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

# Endpoint untuk mengecek apakah server berjalan
@app.get("/")
def read_root():
    return {"message": "Server is running"}

# Contoh request body
class PredictionRequest(BaseModel):
    data: list

# Endpoint untuk memprediksi data (sebagai tensor)
@app.post("/predict_data/")
def predict_data(request: PredictionRequest):
    try:
        # Konversi data input ke dalam tensor
        input_data = tf.constant(request.data, dtype=tf.float32)
        # Lakukan prediksi menggunakan model
        predictions = model(input_data).numpy().tolist()
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint untuk memprediksi kelas dari gambar yang diunggah
@app.post("/predict_image/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Baca file gambar yang diunggah
        contents = await file.read()
        image = tf.image.decode_image(contents, channels=3)
        # Ubah ukuran gambar menjadi 224x224 sesuai ekspektasi model
        image = tf.image.resize(image, [224, 224])
        image = tf.expand_dims(image, axis=0)  # Tambahkan batch dimension
        image = tf.cast(image, tf.float32) / 255.0  # Normalisasi

        # Lakukan prediksi
        predictions = model.predict(image)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class_name = classes_names[predicted_class_idx]
        probability = predictions[0][predicted_class_idx]

        # Cek probabilitas
        if probability >= 0.8:
            result = {"class": predicted_class_name, "probability": float(probability)}
        else:
            result = {"class": "barang tidak diketahui", "probability": float(probability)}

        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Menjalankan server (gunakan perintah ini hanya untuk local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
