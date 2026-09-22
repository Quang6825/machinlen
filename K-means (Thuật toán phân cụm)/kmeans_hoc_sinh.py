from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

root = Path(__file__).resolve().parent
csv_path = root / "hoc_sinh_du_lieu.csv"
tab_path = root / "hoc_sinh_du_lieu_orange.tab"


def load_data():
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if tab_path.exists():
        return pd.read_csv(tab_path, sep='\t')
    raise FileNotFoundError("Không tìm thấy file dữ liệu .csv hoặc .tab trong thư mục hiện tại.")


df = load_data()
print("Dữ liệu đã đọc:")
print(df)

X = df[["HoursPerDay", "ExamScore"]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nDữ liệu sau khi chuẩn hóa:")
print(pd.DataFrame(X_scaled, columns=["HoursPerDay_scaled", "ExamScore_scaled"]))

inertia = []
for k in range(1, 6):
    model = KMeans(n_clusters=k, n_init=10, random_state=0)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

print("\nGiá trị Elbow Method (WCSS):")
for k, wcss in enumerate(inertia, start=1):
    print(f"k={k}, WCSS={wcss:.6f}")

plt.figure(figsize=(6, 4))
plt.plot(range(1, 6), inertia, marker='o', linestyle='--')
plt.xlabel('k')
plt.ylabel('WCSS / Inertia')
plt.title('Elbow Method for K-means')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(root / 'elbow_kmeans.png', dpi=150)
print("\nĐã lưu biểu đồ Elbow tại: elbow_kmeans.png")

k = 2
model = KMeans(n_clusters=k, n_init=10, random_state=0)
labels = model.fit_predict(X_scaled)

cluster_summary = df.copy()
cluster_summary["Cluster"] = labels

print("\nKết quả phân cụm K-means (k=2):")
print(cluster_summary[["Student", "HoursPerDay", "ExamScore", "Cluster"]])

centers = model.cluster_centers_
print("\nTâm cụm trên dữ liệu chuẩn hóa:")
for idx, center in enumerate(centers, start=1):
    print(f"- Cluster {idx}: HoursPerDay={center[0]:.2f}, ExamScore={center[1]:.2f}")

print("\nNhóm học sinh theo cluster:")
for cluster_id in range(k):
    members = cluster_summary.loc[cluster_summary["Cluster"] == cluster_id, "Student"].tolist()
    print(f"- Cluster {cluster_id}: {members}")
