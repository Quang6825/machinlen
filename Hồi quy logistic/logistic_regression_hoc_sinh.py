import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Dữ liệu học sinh

df = pd.DataFrame({
    'gio_hoc': [1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
    'diem_kiem_tra': [4, 5, 5, 6, 6, 7, 7, 8, 9, 9],
    'dau': [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
})

X = df[['gio_hoc', 'diem_kiem_tra']]
y = df['dau']

model = LogisticRegression(max_iter=10000)
model.fit(X, y)

# Dự đoán trên tập dữ liệu
pred = model.predict(X)
print('Hệ số beta_0:', model.intercept_[0])
print('Hệ số beta_1 (gio_hoc):', model.coef_[0][0])
print('Hệ số beta_2 (diem_kiem_tra):', model.coef_[0][1])
print('Độ chính xác:', accuracy_score(y, pred))
print('Dự đoán:', pred.tolist())

# Xác suất đậu cho từng mẫu
print('\nXác suất đậu theo từng học sinh:')
for i, row in df.iterrows():
    p = model.predict_proba([[row['gio_hoc'], row['diem_kiem_tra']]])[0, 1]
    print(f"HS {i+1}: gio_hoc={row['gio_hoc']}, diem={row['diem_kiem_tra']}, P(dau)={p:.4f}, label={row['dau']}")

# Ví dụ học sinh mới
new_student = pd.DataFrame({'gio_hoc': [4], 'diem_kiem_tra': [7]})
print('\nXác suất học sinh mới đậu:', model.predict_proba(new_student)[0, 1])
