import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="SuperKart Sales Predictor", page_icon=":shopping_trolley:")

# 'backend' is the container name of the Flask API on the shared Docker network (see deployment instructions)
BACKEND_URL = "http://backend:7860"

st.title("SuperKart - Sales Forecast")
st.write("Predict the total sales revenue for a product-store combination.")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

# ---------------------------------------------------------------------
# Tab 1: Online inference - a single prediction
# ---------------------------------------------------------------------
with tab1:
    st.subheader("Enter Product & Store Details")

    col1, col2 = st.columns(2)
    with col1:
        product_weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
        product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, value=0.027, format="%.3f")
        product_mrp = st.number_input("Product MRP", min_value=0.0, value=117.08)
        product_id_char = st.selectbox("Product Category Code (FD=Food, DR=Drinks, NC=Non-Consumable)", ["FD", "DR", "NC"])
    with col2:
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"])
        store_age_years = st.number_input("Store Age (Years)", min_value=0, value=16)
        product_type_category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

    if st.button("Predict Sales"):
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": product_sugar_content,
            "Product_Allocated_Area": product_allocated_area,
            "Product_MRP": product_mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": store_location_city_type,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Store_Age_Years": store_age_years,
            "Product_Type_Category": product_type_category,
        }
        try:
            response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted Sales: {result['predicted_sales']}")
            else:
                st.error(f"Error from API: {response.text}")
        except Exception as e:
            st.error(f"Could not reach the backend API: {e}")

# ---------------------------------------------------------------------
# Tab 2: Batch inference - upload a CSV of many rows
# ---------------------------------------------------------------------
with tab2:
    st.subheader("Upload CSV for Batch Prediction")
    st.caption(
        "The CSV must contain these columns: Product_Weight, Product_Sugar_Content, "
        "Product_Allocated_Area, Product_MRP, Store_Size, Store_Location_City_Type, "
        "Store_Type, Product_Id_char, Store_Age_Years, Product_Type_Category"
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        preview_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(preview_df.head())

        if st.button("Run Batch Prediction"):
            uploaded_file.seek(0)
            files = {"file": uploaded_file.getvalue()}
            try:
                response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files=files, timeout=30)
                if response.status_code == 200:
                    predictions = response.json()
                    preview_df["Predicted_Sales"] = preview_df.index.astype(str).map(predictions)
                    st.success("Batch prediction complete!")
                    st.dataframe(preview_df)
                    csv_out = preview_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Predictions as CSV", csv_out, "superkart_predictions.csv", "text/csv")
                else:
                    st.error(f"Error from API: {response.text}")
            except Exception as e:
                st.error(f"Could not reach the backend API: {e}")
