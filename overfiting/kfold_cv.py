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

# Chạy K-Fold Cross Validation (K = 5)
K = 5
indices = np.arange(len(X))
np.random.seed(42)
np.random.shuffle(indices)
folds = np.array_split(indices, K)

degrees_to_test = list(range(1, 16))
avg_cv_mse_list = []

for d in degrees_to_test:
    mse_folds = []
    for i in range(K):
        val_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(K) if j != i])

        X_tr, y_tr = X_norm[train_idx], y[train_idx]
        X_va, y_va = X_norm[val_idx], y[val_idx]

        X_tr_p = np.vander(X_tr, d + 1, increasing=True)
        X_va_p = np.vander(X_va, d + 1, increasing=True)

        w_d = np.linalg.pinv(X_tr_p.T @ X_tr_p) @ X_tr_p.T @ y_tr
        y_va_pred = X_va_p @ w_d
        mse_folds.append(np.mean((y_va_pred - y_va)**2))

    avg_cv_mse_list.append(np.mean(mse_folds))

best_degree = degrees_to_test[np.argmin(avg_cv_mse_list)]

print("=== K-FOLD CROSS VALIDATION (K=5) ===")
print(f"-> Bậc đa thức tối ưu chọn bởi K-Fold: BẬC {best_degree}")
print(f"-> MSE Validation trung bình: {min(avg_cv_mse_list):.4f}\n")

# Huấn luyện lại mô hình Bậc 15 và Bậc tối ưu để vẽ đồ thị so sánh
degree_overfit = 15

# Mô hình Bậc 15
X_poly_overfit = np.vander(X_norm, degree_overfit + 1, increasing=True)
w_overfit = np.linalg.pinv(X_poly_overfit.T @ X_poly_overfit) @ X_poly_overfit.T @ y

# Mô hình tối ưu
X_poly_best = np.vander(X_norm, best_degree + 1, increasing=True)
w_best = np.linalg.pinv(X_poly_best.T @ X_poly_best) @ X_poly_best.T @ y

# Vẽ đồ thị so sánh
X_dense = np.linspace(X.min() - 5, X.max() + 5, 300)
X_dense_norm = (X_dense - X_mean) / X_std

y_dense_overfit = np.vander(X_dense_norm, degree_overfit + 1, increasing=True) @ w_overfit
y_dense_best = np.vander(X_dense_norm, best_degree + 1, increasing=True) @ w_best

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='black', label='Dữ liệu thực tế', zorder=5)
plt.plot(X_dense, y_dense_overfit, color='red', linestyle='--', linewidth=2, label=f'Overfitting (Bậc {degree_overfit})')
plt.plot(X_dense, y_dense_best, color='green', linewidth=2.5, label=f'K-Fold CV chọn (BẬC {best_degree} - Tối ưu)')

plt.title('So sánh Mô hình Overfitting vs Mô hình tối ưu chọn bởi K-Fold', fontsize=13, fontweight='bold')
plt.xlabel('Diện tích (m²)')
plt.ylabel('Giá nhà (Triệu VNĐ)')
plt.ylim(y.min() - 20, y.max() + 30)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()