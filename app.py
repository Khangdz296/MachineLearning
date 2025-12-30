from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
import requests # <--- THÊM MỚI: Thư viện để gọi API Firebase
import json
from xgboost import XGBClassifier

app = Flask(__name__)

# --- CẤU HÌNH FIREBASE ---
FIREBASE_URL = 'https://machinelearning-97312-default-rtdb.asia-southeast1.firebasedatabase.app/users.json'

# --- 1. ĐỊNH NGHĨA HÀM FEATURE ENGINEERING (Copy từ Notebook sang) ---
def feature_engineering(df):
    df_out = df.copy()
    
    # 1. Tạo chỉ số rủi ro đường huyết
    if 'blood_glucose_level' in df_out.columns and 'HbA1c_level' in df_out.columns:
        df_out['sugar_risk_score'] = df_out['blood_glucose_level'] * df_out['HbA1c_level']
    
    # 2. Tạo nhóm rủi ro bệnh nền
    if 'hypertension' in df_out.columns and 'heart_disease' in df_out.columns:
        df_out['has_condition'] = df_out[['hypertension', 'heart_disease']].max(axis=1)
        
    return df_out

# --- 2. LOAD MODEL VÀ CÁC THÀNH PHẦN BẰNG PICKLE ---
try:
    with open('diabetes_deploy_model.pkl', 'rb') as f: # Mode 'rb' - read binary
        data = pickle.load(f)
    print("Các thành phần có trong file:", data['features_kept'])
    
    model = data['model']
    preprocessor = data['preprocessor']
    final_scaler = data['final_scaler']    # Load cái Scaler quan trọng
    features_kept = data['features_kept']  # Load danh sách cột chuẩn
    
    print(">> Load model bằng Pickle thành công!")
except Exception as e:
    print(f">> LỖI LOAD MODEL: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return "Lỗi: Server chưa load được Model."

    try:
        # --- 3. LẤY DỮ LIỆU TỪ FORM ---
        raw_gender = request.form['gender']
        raw_age = float(request.form['age'])
        raw_hypertension = int(request.form['hypertension'])
        raw_heart_disease = int(request.form['heart_disease'])
        raw_smoking = request.form['smoking_history']
        raw_bmi = float(request.form['bmi'])
        raw_hba1c = float(request.form['HbA1c_level'])
        raw_glucose = float(request.form['blood_glucose_level'])

        # Tạo DataFrame thô
        input_data_df = pd.DataFrame({
            'gender': [raw_gender],
            'age': [raw_age],
            'hypertension': [raw_hypertension],
            'heart_disease': [raw_heart_disease],
            'smoking_history': [raw_smoking],
            'bmi': [raw_bmi],
            'HbA1c_level': [raw_hba1c],
            'blood_glucose_level': [raw_glucose] 
        })
        
        # --- 4. PIPELINE XỬ LÝ (Chuẩn 100% theo Notebook) ---
        
        # B1: Tiền xử lý (Impute + OneHot)
        X_clean = preprocessor.transform(input_data_df)
        
        # B2: Feature Engineering (Tạo biến mới)
        X_eng = feature_engineering(X_clean)
        
        # B3: Lựa chọn cột (Selection) - Giữ lại đúng cột model cần
        X_selected = X_eng[features_kept]
        
        # B4: Chuẩn hóa (Scaling) - Đưa về Z-score
        X_final = final_scaler.transform(X_selected)

        # B5: Dự báo
        prediction = model.predict(X_final)
        output = int(prediction[0])
        # --- 5. GỬI LOG LÊN FIREBASE ---
        try:
            log_data = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'age': raw_age,
                'gender': raw_gender,
                'bmi': raw_bmi,
                'hypertension': raw_hypertension,
                'heart_disease': raw_heart_disease,
                'smoking_history': raw_smoking,
                'HbA1c_level': raw_hba1c,
                'blood_glucose_level': raw_glucose,
                'prediction_result': output
            }
            # Gửi request (bỏ qua xác thực SSL nếu cần thiết, nhưng tốt nhất nên để mặc định)
            requests.post(FIREBASE_URL, json=log_data)
            print(">> Da gui du lieu len Firebase!")
        except Exception as err:
            print(f"Loi Firebase: {err}")

        # --- 6. TRẢ KẾT QUẢ ---
        if output == 1:
            text = "CẢNH BÁO: Nguy cơ cao bị tiểu đường"
            color = "#dc3545"
        else:
            text = "An toàn: Nguy cơ thấp"
            color = "#28a745"

        return render_template('index.html', prediction_text=text, color=color)

    except Exception as e:
        print(f"Loi runtime: {e}")
        return render_template('index.html', prediction_text=f"Có lỗi xảy ra: {str(e)}", color="orange")

if __name__ == "__main__":
    app.run(debug=True)