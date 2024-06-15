# Gunakan image dasar dari Python 3.9
FROM python:3.9-slim

# Atur direktori kerja
WORKDIR /app

# Salin file requirements.txt ke direktori kerja
COPY requirements.txt .

# Install dependensi dari requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh isi proyek ke dalam direktori kerja
COPY . .

# Ekspose port 8000
EXPOSE 8000

# Jalankan aplikasi menggunakan Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
