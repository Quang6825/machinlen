import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression

DATA_FILE = Path(__file__).with_name("hoc_sinh_du_lieu.csv")


def main():
    df = pd.read_csv(DATA_FILE)
    X = df[["HoursPerDay", "ExamScore"]]
    y = df["PassNumeric"]

    model = LinearRegression()
    model.fit(X, y)

    print("=== Dữ liệu học sinh ===")
    print(df.to_string(index=False))
    print()

    print("=== Hồi quy tuyến tính ===")
    print(f"b0 = {model.intercept_:.6f}")
    print(f"b1 (HoursPerDay) = {model.coef_[0]:.6f}")
    print(f"b2 (ExamScore) = {model.coef_[1]:.6f}")
    print(f"Y = {model.intercept_:.4f} + {model.coef_[0]:.4f}*HoursPerDay + {model.coef_[1]:.4f}*ExamScore")
    print()

    print("=== Dự đoán ===")
    for _, row in df.iterrows():
        x_input = pd.DataFrame(
            [{"HoursPerDay": row["HoursPerDay"], "ExamScore": row["ExamScore"]}]
        )
        y_pred = model.predict(x_input)[0]
        print(
            f"{row['Student']} | Giờ={row['HoursPerDay']} | Điểm={row['ExamScore']} | "
            f"Thực tế={row['PassNumeric']} | Dự đoán={y_pred:.4f}"
        )


if __name__ == "__main__":
    main()
