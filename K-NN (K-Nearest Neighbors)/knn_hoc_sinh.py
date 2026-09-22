import sys
sys.stdout.reconfigure(encoding='utf-8')
import csv
import math
from typing import List, Dict, Tuple

DATA_FILE = r"D:\Quang code\AI\machinlen\machinlen\K-NN (K-Nearest Neighbors)\hoc_sinh_du_lieu.csv"


def load_dataset(path: str) -> List[Dict[str, object]]:
    """Đọc dữ liệu từ file CSV."""
    rows: List[Dict[str, object]] = []
    with open(path, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append({
                "Student": row["Student"],
                "HoursPerDay": float(row["HoursPerDay"]),
                "ExamScore": float(row["ExamScore"]),
                "PassStatus": row["PassStatus"],
                "PassNumeric": int(row["PassNumeric"]),
            })
    return rows

def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def knn_predict(train_data: List[Dict[str, object]], sample: Tuple[float, float], k: int = 3) -> str:
    """Dự đoán nhãn theo nguyên tắc KNN với khoảng cách Euclid."""
    if k <= 0:
        raise ValueError("k phải lớn hơn 0")

    distances = []
    for row in train_data:
        point = (float(row["HoursPerDay"]), float(row["ExamScore"]))
        dist = euclidean_distance(point, sample)
        distances.append((dist, row["PassStatus"]))

    distances.sort(key=lambda item: item[0])
    neighbors = distances[:k]

    vote_map = {}
    for _, label in neighbors:
        vote_map[label] = vote_map.get(label, 0) + 1

    predicted_label = max(vote_map.items(), key=lambda item: (item[1], item[0]))[0]
    return predicted_label


def evaluate_knn(data: List[Dict[str, object]], k: int = 3) -> Tuple[float, List[Tuple[str, str, str]]]:
    """Đánh giá độ chính xác bằng cách bỏ một mẫu ra khỏi tập huấn luyện."""
    matches = []
    correct = 0
    total = len(data)

    for row in data:
        sample = (float(row["HoursPerDay"]), float(row["ExamScore"]))
        training_set = [item for item in data if item["Student"] != row["Student"]]
        predicted = knn_predict(training_set, sample, k=k)
        actual = str(row["PassStatus"])
        matches.append((row["Student"], actual, predicted))
        if predicted == actual:
            correct += 1

    accuracy = correct / total if total else 0.0
    return accuracy, matches


def main() -> None:
    data = load_dataset(DATA_FILE)

    print("\n=== Dữ liệu học sinh ===")
    for row in data:
        print(f"{row['Student']}: HoursPerDay={row['HoursPerDay']}, ExamScore={row['ExamScore']}, PassStatus={row['PassStatus']}")

    for k in [1, 3, 5]:
        accuracy, matches = evaluate_knn(data, k=k)
        print(f"\n=== K = {k} ===")
        print(f"Độ chính xác: {accuracy * 100:.2f}%")
        for student, actual, predicted in matches:
            print(f"{student}: thực tế={actual}, dự đoán={predicted}")

    new_student = (4.5, 78)
    print(f"\n=== Dự đoán cho học sinh mới ===")
    print(f"Input: HoursPerDay={new_student[0]}, ExamScore={new_student[1]}")
    pred = knn_predict(data, new_student, k=3)
    print(f"Kết quả dự đoán: {pred}")


if __name__ == "__main__":
    main()
