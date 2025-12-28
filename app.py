from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

app = Flask(__name__)

# 1. LOAD MODEL VÀ CÁC THÀNH PHẦN
try:
    with open('diabetes_deploy_model.pkl', 'rb') as f:
        data = pickle.load(f)
    
    model = data['model']
    preprocessor = data['preprocessor']
    feature_names_full = data['features_after_preprocessing'] # Tên cột đầy đủ
    features_kept = data['features_kept'] # Tên cột cần giữ lại
    
    print("Load model thanh cong!")
except Exception as e:
    print(f"Loi load model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return "Loi: Chua co Model."

    try:
        # 2. LẤY DỮ LIỆU TỪ FORM (TẠO DATAFRAME THÔ)
        # Lưu ý: Tên cột ở đây phải KHỚP Y HỆT với tên cột trong biến X ban đầu của bạn
        input_data = {
            'gender': [request.form['gender']],
            'age': [float(request.form['age'])],
            'hypertension': [int(request.form['hypertension'])],
            'heart_disease': [int(request.form['heart_disease'])],
            'smoking_history': [request.form['smoking_history']],
            'bmi': [float(request.form['bmi'])],
            'HbA1c_level': [float(request.form['HbA1c_level'])],
            'blood_glucose_level': [float(request.form['blood_glucose_level'])] 
            # Lưu ý: blood_glucose_level trong training là numerical, nên để float cho chắc
        }
        
        df_raw = pd.DataFrame(input_data)

        # 3. TIỀN XỬ LÝ (PREPROCESSING)
        # Dùng preprocessor đã lưu để transform. Nó sẽ tự Scale và OneHot.
        # Kết quả trả về là một mảng Numpy (mất tên cột)
        processed_array = preprocessor.transform(df_raw)

        # 4. TÁI TẠO DATAFRAME (ĐỂ LỌC CỘT)
        # Gán lại tên cột cho mảng numpy vừa tạo
        df_processed = pd.DataFrame(processed_array, columns=feature_names_full)

        # 5. LỌC ĐẶC TRƯNG (FEATURE SELECTION)
        # Chỉ giữ lại các cột mà mô hình đã được học (bước lọc tương quan)
        df_final = df_processed[features_kept]

        # 6. DỰ ĐOÁN
        prediction = model.predict(df_final)
        output = prediction[0]
        if output == 1:
            text = "Canh bao: Nguy co cao bi tieu đuong"
            color = "#dc3545"
        else:
            text = "An toan: Nguy co thap"
            color = "#28a745"

        return render_template('index.html', prediction_text=text, color=color)

    except Exception as e:
        print(f"Loi chi tiet: {e}")
        return render_template('index.html', prediction_text=f"Co loi: {str(e)}", color="orange")

if __name__ == "__main__":
    app.run(debug=True)