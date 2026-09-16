import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Đọc và chuẩn hóa dữ liệu
filename = 'data_linear.csv'
df = pd.read_csv(filename)
X = df.iloc[:, 0].values.astype(float)
y = df.iloc[:, 1].values.astype(float)

X_mean, X_std = np.mean(X), np.std(X)
X_norm = (X - X_mean) / X_std

# Huấn luyện mô hình đa thức Bậc 15 (Overfitting)
degree = 15
X_poly = np.vander(X_norm, degree + 1, increasing=True)
w = np.linalg.pinv(X_poly.T @ X_poly) @ X_poly.T @ y

y_pred = X_poly @ w
mse = np.mean((y_pred - y)**2)

print("=== MÔ HÌNH OVERFITTING (BẬC 15) ===")
print(f"-> MSE trên tập huấn luyện: {mse:.4f}")

# Vẽ biểu đồ hiển thị hiện tượng Overfitting
X_dense = np.linspace(X.min() - 5, X.max() + 5, 300)
X_dense_norm = (X_dense - X_mean) / X_std
X_dense_poly = np.vander(X_dense_norm, degree + 1, increasing=True)
y_dense_pred = X_dense_poly @ w

plt.figure(figsize=(9, 5))
plt.scatter(X, y, color='black', label='Dữ liệu thực tế', zorder=5)
plt.plot(X_dense, y_dense_pred, color='red', linestyle='--', linewidth=2, label=f'Overfitting (Bậc {degree})')

plt.title('Minh họa Hiện tượng Overfitting (Bậc 15)', fontsize=12, fontweight='bold')
plt.xlabel('Diện tích (m²)')
plt.ylabel('Giá nhà (Triệu VNĐ)')
plt.ylim(y.min() - 20, y.max() + 30)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()