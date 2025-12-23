import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# --- PHẦN QUAN TRỌNG: LOAD FILE HỆ THỐNG ---
filename = "Model_AI.pkl"

try:
    with open(filename, "rb") as f:
        saved_data = pickle.load(f)
    # SỬA LẠI DÒNG NÀY ĐỂ KHÔNG BỊ LỖI FONT TRÊN WINDOWS
    print(">> Da load file he thong thanh cong! San sang du doan.") 
except FileNotFoundError:
    print(f">> LOI: Khong tim thay file '{filename}'.")
    exit()

# Lấy dữ liệu ra
model = saved_data["model"]
maps = saved_data["mappings"]
threshold = saved_data["threshold"]

scaler_age = saved_data["scaler_age"]
scaler_glucose = saved_data["scaler_glucose"]
scaler_bmi = saved_data["scaler_bmi"]
scaler_work = saved_data["scaler_work"]
scaler_smoke = saved_data["scaler_smoke"]

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = {
            'gender': request.form['gender'],
            'age': float(request.form['age']),
            'hypertension': int(request.form['hypertension']),
            'heart_disease': int(request.form['heart_disease']),
            'ever_married': request.form['ever_married'],
            'work_type': request.form['work_type'],
            'Residence_type': request.form['Residence_type'],
            'avg_glucose_level': float(request.form['avg_glucose_level']),
            'bmi': float(request.form['bmi']),
            'smoking_status': request.form['smoking_status']
        }

        # Xử lý (Encoding)
        input_data['gender'] = maps['gender'][input_data['gender']]
        input_data['ever_married'] = maps['ever_married'][input_data['ever_married']]
        input_data['work_type'] = maps['work_type'][input_data['work_type']]
        input_data['Residence_type'] = maps['Residence_type'][input_data['Residence_type']]
        input_data['smoking_status'] = maps['smoking_status'][input_data['smoking_status']]

        df_new = pd.DataFrame([input_data])

        # Chuẩn hóa (Scaling)
        df_new['age'] = scaler_age.transform(df_new[['age']])
        df_new['avg_glucose_level'] = scaler_glucose.transform(df_new[['avg_glucose_level']])
        df_new['bmi'] = scaler_bmi.transform(df_new[['bmi']])
        df_new['work_type'] = scaler_work.transform(df_new[['work_type']])
        df_new['smoking_status'] = scaler_smoke.transform(df_new[['smoking_status']])

        cols_order = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
                      'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
        final_features = df_new[cols_order]

        # Dự đoán
        y_prob = model.predict_proba(final_features)[:, 1]
        percent = y_prob[0] * 100
        
        if y_prob[0] >= threshold:
            # Chữ hiển thị trên web (HTML) thì có dấu thoải mái
            res_text = f"CẢNH BÁO: Nguy cơ Đột quỵ CAO! (Tỷ lệ: {percent:.1f}%)"
        else:
            res_text = f"AN TOÀN: Nguy cơ thấp. (Tỷ lệ: {percent:.1f}%)"

        return render_template("index.html", prediction_text=res_text)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Lỗi: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)