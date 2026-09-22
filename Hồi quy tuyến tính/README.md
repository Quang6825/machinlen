# Hồi quy tuyến tính cho dữ liệu học sinh

## Dữ liệu

Tệp dữ liệu gốc:
- `hoc_sinh_du_lieu.csv`
- `hoc_sinh_du_lieu_orange.tab`

Các cột quan trọng:
- `HoursPerDay`: số giờ học/ngày
- `ExamScore`: điểm bài kiểm tra
- `PassNumeric`: kết quả nhị phân (0 = Không, 1 = Có)

## Mô hình

Ta dùng hồi quy tuyến tính với 2 biến đầu vào:
- $X_1$ = số giờ học
- $X_2$ = điểm bài kiểm tra

Mục tiêu:
- $Y$ = mức độ đậu ước lượng theo số 0 đến 1

Công thức:

$$
Y = b_0 + b_1 X_1 + b_2 X_2
$$

## Chạy Python

```bash
python linear_regression_hoc_sinh.py
```

## Mở trong Orange

1. Mở Orange.
2. Chọn File > Open Data Table.
3. Mở `hoc_sinh_du_lieu_orange.tab`.
4. Thêm widget `Linear Regression`.
5. Chọn:
   - Features: `HoursPerDay`, `ExamScore`
   - Target: `PassNumeric`

## Kết luận

Đây là bài toán hồi quy tuyến tính vì ta đang dự đoán một giá trị số (mức độ đậu) từ hai biến đầu vào. Đây là cách phù hợp nhất để áp dụng thuật toán hồi quy tuyến tính vào bài toán này.
