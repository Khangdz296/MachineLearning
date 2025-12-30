import pickle

with open('diabetes_deploy_model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
features_kept = data['features_kept']

print("1. Danh sách bạn lưu (Features Kept):")
print(features_kept)
print(f"-> Số lượng: {len(features_kept)}")

print("\n2. Danh sách Model thực sự cần (Feature Names In):")
# Model XGBoost lưu tên cột nó đã học trong thuộc tính này
try:
    print(model.feature_names_in_)
    print(f"-> Số lượng: {len(model.feature_names_in_)}")
except:
    print("Model không lưu tên cột (nhưng chắc chắn là khác danh sách trên).")

# So sánh
if len(features_kept) != len(model.feature_names_in_):
    print("\n=> KẾT LUẬN: LỆCH NHAU RỒI! Model học ít cột hơn danh sách bạn lưu.")