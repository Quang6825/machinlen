import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report

# Đọc dữ liệu
file_path = r"D:\Quang code\AI\machinlen\machinlen\Cây quyết định\hoc_sinh_du_lieu.csv"
df = pd.read_csv(file_path)

# Chuẩn bị dữ liệu
X = df[["HoursPerDay", "ExamScore"]]
y = df["PassStatus"]

# Huấn luyện mô hình cây quyết định
model = DecisionTreeClassifier(random_state=42, max_depth=3)
model.fit(X, y)

# Dự đoán trên cùng tập dữ liệu
pred = model.predict(X) 

# Đánh giá mô hình
accuracy = accuracy_score(y, pred)
print("Độ chính xác trên tập dữ liệu huấn luyện:", round(accuracy, 4))
print("\nBáo cáo phân loại:")
print(classification_report(y, pred, zero_division=0))

# Hiển thị cấu trúc cây
print("\nCây quyết định:")
print(export_text(model, feature_names=["HoursPerDay", "ExamScore"]))

# Dự đoán cho học sinh mới
new_student = pd.DataFrame({
    "HoursPerDay": [4],
    "ExamScore": [72]
})
result = model.predict(new_student)
print("\nDự đoán cho học sinh mới (HoursPerDay=4, ExamScore=72):", result[0])
