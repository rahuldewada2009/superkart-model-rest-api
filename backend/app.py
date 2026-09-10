from flask import Flask, request, jsonify
import pandas as pd
import joblib

# loading the trained pipeline (preprocessing + model bundled together)
model = joblib.load("superkart_model.joblib")

# the exact set of features the model pipeline expects, in this order
FEATURE_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]

superkart_api = Flask(__name__)


@superkart_api.get("/")
def home():
    # simple health-check endpoint
    return jsonify({"message": "SuperKart Sales Prediction API is up and running."})


@superkart_api.post("/v1/predict")
def predict():
    # online / single-record inference
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    missing_cols = [col for col in FEATURE_COLUMNS if col not in data]
    if missing_cols:
        return jsonify({"error": f"Missing required field(s): {missing_cols}"}), 400

    try:
        input_df = pd.DataFrame([data], columns=FEATURE_COLUMNS)
        prediction = model.predict(input_df)[0]
    except Exception as e:
        return jsonify({"error": f"Could not generate a prediction: {str(e)}"}), 400

    return jsonify({"predicted_sales": round(float(prediction), 2)})


@superkart_api.post("/v1/predictbatch")
def predict_batch():
    # batch inference - a CSV file is uploaded under the form field named 'file'
    if "file" not in request.files:
        return jsonify({"error": "No file part named 'file' found in the request."}), 400

    file = request.files["file"]

    try:
        input_df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Could not read the uploaded CSV file: {str(e)}"}), 400

    missing_cols = [col for col in FEATURE_COLUMNS if col not in input_df.columns]
    if missing_cols:
        return jsonify({"error": f"Missing required column(s) in CSV: {missing_cols}"}), 400

    try:
        predictions = model.predict(input_df[FEATURE_COLUMNS])
    except Exception as e:
        return jsonify({"error": f"Could not generate predictions: {str(e)}"}), 400

    # returning predictions keyed by row index, so the frontend can map them back to the input rows
    result = {str(i): round(float(pred), 2) for i, pred in enumerate(predictions)}
    return jsonify(result)


if __name__ == "__main__":
    # port 7860 is the port we will expose in the Dockerfile below
    superkart_api.run(host="0.0.0.0", port=7860)
