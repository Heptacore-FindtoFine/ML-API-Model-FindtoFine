# FindtoFine ML API Model for Bangkit Capstone Project

## Prequisite
- Python

## Getting Started
### Step 1 : Clone Repo
```
git clone https://github.com/Heptacore-FindtoFine/ML-API-Model-FindtoFine.git
```
### Step 2 : Download all dependencies
```
pip install --no-cache-dir -r requirements.txt
```
### Step 3 : Run the server
```
uvicorn main:app
```

## API SPEC
### API URL
https://ml-model-api-izoaerx5sa-et.a.run.app

### ML API
- Method: `POST`
- Endpoint: `/predict`
- Request:
  - Body:
    ```
      "image_url": "string (link from image url)"
    ```
- Response Success:
  - Status code: `400`
  - Body:
    ```
    {
      "predicted_class": "string (item name)",
      "probability": "float"
    }
    ```
- Response Error:
  - Status code: `500`
  - Body:
    ```
    {
      "detail": "Error downloading image: Invalid URL `url`: No scheme supplied. Perhaps you     meant `url`"
    }
    ```
