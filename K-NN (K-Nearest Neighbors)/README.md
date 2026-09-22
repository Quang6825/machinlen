# Thuật toán k-NN trên dữ liệu học sinh

Tập dữ liệu trong file [hoc_sinh_du_lieu.csv](hoc_sinh_du_lieu.csv) chứa các thuộc tính:
- Student: tên học sinh
- HoursPerDay: số giờ học mỗi ngày
- ExamScore: điểm thi
- PassStatus: trạng thái đậu/rớt
- PassNumeric: nhãn số (0/1)

File [knn_hoc_sinh.py](knn_hoc_sinh.py) triển khai thuật toán k-NN theo khoảng cách Euclidean để dự đoán trạng thái đậu/rớt.

## Cách chạy

```bash
cd "d:\machinlen\K-NN (K-Nearest Neighbors)"
C:/Users/Admin/AppData/Local/Programs/Python/Python312/python.exe knn_hoc_sinh.py
```

## Kết quả mẫu

- Với k = 1: độ chính xác 100%
- Với k = 3: độ chính xác 90%
- Với k = 5: độ chính xác 70%
- Học sinh mới có HoursPerDay = 4.5, ExamScore = 78 dự đoán là "Co"
