# SuperKart Sales Forecast - Deployment

This repository holds the deployment code for my SuperKart Sales Forecasting capstone project. It packages the trained regression model (Random Forest / XGBoost pipeline, selected by test RMSE) behind a Flask REST API, with a Streamlit UI on top for online and batch inference. Both services run as separate Docker containers connected over a shared Docker network, deployed here through GitHub Codespaces.

Model training, EDA, and evaluation live in the project notebook (`SuperKart_Model_Deployment_Solution.ipynb`), not in this repo — this repo is only the two deployable services.

## Structure

```
backend/                 # Flask inference API
  app.py
  requirements.txt
  Dockerfile
  superkart_model.joblib # serialized sklearn Pipeline (preprocessing + model)
frontend/                # Streamlit UI
  app.py
  requirements.txt
  Dockerfile
```

## API

- `GET /` - health check
- `POST /v1/predict` - single (online) prediction, JSON body with the 10 model features
- `POST /v1/predictbatch` - batch prediction, multipart form file upload (`file`) with a CSV of the same 10 columns

Required feature columns: `Product_Weight`, `Product_Sugar_Content`, `Product_Allocated_Area`, `Product_MRP`, `Store_Size`, `Store_Location_City_Type`, `Store_Type`, `Product_Id_char`, `Store_Age_Years`, `Product_Type_Category`.

## Running in a GitHub Codespace

```bash
# build both images
cd backend  && docker build -t superkart-backend .  && cd ..
cd frontend && docker build -t superkart-frontend . && cd ..

# shared network so the containers can reach each other by name
docker network create superkart-network

# run both containers
docker run -d --name backend  --network superkart-network -p 7860:7860 superkart-backend
docker run -d --name frontend --network superkart-network -p 8501:8501 superkart-frontend
```

Then, in the Codespace **Ports** tab, forward ports `7860` (backend) and `8501` (frontend) and set both to **Public**. Open the forwarded URL for `8501` to use the Streamlit app; the forwarded URL for `7860` is the backend API root used for direct/online inference from the notebook.
