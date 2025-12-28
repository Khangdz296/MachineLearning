from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier

# MÃ HÓA VÀ CHUẨN HÓA
X = df.drop('diabetes', axis=1)
y = df['diabetes']

numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_cols = ['gender', 'smoking_history']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

X_processed = preprocessor.fit_transform(X)

new_cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_feature_names = numerical_cols + list(new_cat_names) + ['hypertension', 'heart_disease']

X_final_df = pd.DataFrame(X_processed, columns=all_feature_names)

y = y.reset_index(drop=True)

final_df = pd.concat([X_final_df, y], axis=1)

final_df.to_csv('diabetes_processed_full.csv', index=False)

print("Xử lý xong! File output: diabetes_processed_full.csv")
final_df.tail()



# LỰA CHỌN ĐẶC TRƯNG
data = pd.read_csv("diabetes_processed_full.csv")

corr = data.corr()

target_corr = corr['diabetes'].abs().sort_values(ascending=False)
print("Độ tương quan với biến mục tiêu:\n", target_corr)

threshold = 0.05
features_to_drop = target_corr[target_corr < threshold].index.tolist()

print(f"\nCác cột sẽ bị loại bỏ (do tương quan < {threshold}):")
print(features_to_drop)

data_selected = data.drop(columns=features_to_drop)

X_final = data_selected.drop('diabetes', axis=1)
y_final = data_selected['diabetes']

print("\nĐã lọc xong! Số lượng đặc trưng còn lại:", X_final.shape[1])

#CHIA DỮ LIỆU

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.3, random_state=42
)

# HUẤN LUYỆN


print("\n--- . HUẤN LUYỆN MÔ HÌNH GRADIENT BOOSTING ---")

print(f"Kích thước tập Train: {X_train.shape}")
print(f"Kích thước tập Test: {X_test.shape}")

xgb_model = XGBClassifier(
    n_estimators=100,     # Số lượng cây
    learning_rate=0.1,    # Tốc độ học
    max_depth=3,          # Độ sâu cây
    subsample=0.8,        # Mỗi cây chỉ học trên 80% dữ liệu ngẫu nhiên -> Chống overfitting cực tốt
    colsample_bytree=0.8, # Mỗi cây chỉ dùng 80% số cột -> Giống Random Forest
    random_state=42,
    eval_metric='logloss' # Thước đo lỗi (Log Loss cho bài toán phân loại)
)

print("Đang huấn luyện... (Có thể mất vài giây)")
xgb_model.fit(X_train, y_train)
# Dự đoán trên tập Test
y_pred = xgb_model.predict(X_train)

print("\n--- ĐÁNH GIÁ KẾT QUẢ ---")
# Độ chính xác tổng thể
acc = accuracy_score(y_train, y_pred)
print(f"Độ chính xác trên tập train (Accuracy): {acc:.4f} ({acc*100:.2f}%)")

# TINH CHỈNH SIÊU THAM SỐ

param_grid = {
    'n_estimators': [100, 200, 300],

    'learning_rate': [0.05, 0.1, 0.2],

    'max_depth': [3, 4, 5],

    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# 2. Thiết lập mô hình cơ sở
xgb_base = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# 3. Khởi tạo Grid Search
grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    n_jobs=-1,
    verbose=1
)

print("-" * 30)
print("Bắt đầu dò tìm bộ tham số tối ưu (Grid Search)...")
grid_search.fit(X_train, y_train)

print("\n--- KẾT QUẢ TỐT NHẤT ---")
print("Bộ tham số vô địch:", grid_search.best_params_)
print(f"Độ chính xác cao nhất trên tập Train (CV): {grid_search.best_score_:.4f}")

print("\n--- SO SÁNH HIỆU QUẢ TRÊN TẬP Train ---")
print(f"Mô hình cũ (Baseline): {acc:.4f}")
print(f"Mô hình sau tinh chỉnh: {grid_search.best_score_:.4f}")

