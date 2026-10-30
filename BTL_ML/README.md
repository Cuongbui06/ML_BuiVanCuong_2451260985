# 4. TRIỂN KHAI VÀ ĐỐI CHIẾU 4 MÔ HÌNH

## 4.1. Linear Regression (Tự cài đặt - OLS Ridge)

* Sử dụng phương trình pháp tuyến (Normal Equation) có thành phần điều chuẩn $L_2$ ($\lambda = 10.0$):
  $$w = (X^T X + \lambda I^*)^{-1} X^T y$$
* Trong đó, ma trận đơn vị $I^*$ được triệt tiêu giá trị tại vị trí bias $(0, 0)$ nhằm không áp đặt phạt lên hệ số chặn $w_0$.
* Điều chuẩn giúp ma trận luôn khả nghịch, tránh hoàn toàn lỗi suy biến do đa cộng tuyến.

## 4.2. Linear Regression (Thư viện - Scikit-Learn `Ridge`)

* Áp dụng mô hình Hồi quy Ridge từ thư viện Scikit-Learn (`sklearn.linear_model.Ridge`).
* Tinh chỉnh tham số `alpha = 10.0` đồng bộ để áp dụng hàm phạt $L_2$ kiểm soát độ lớn của các trọng số.

## 4.3. CART Regression Tree (Tự cài đặt)

* Xây dựng cấu trúc cây nhị phân đệ quy dựa trên độ giảm tổng bình phương sai số (Variance / SSE Reduction).
* Giá trị tại mỗi nút lá là trung bình cộng số học của các mẫu rơi vào lá đó (trên thang đo log).
* **Điều kiện dừng (Stopping criteria):** Cố định độ sâu `max_depth = 5`, ngưỡng số mẫu tối thiểu để chia nhánh `min_samples_split = 20`, và số mẫu tối thiểu ở nút lá `min_samples_leaf = 10`.

## 4.4. CART Regression Tree (Thư viện - Scikit-Learn `DecisionTreeRegressor`)

* Thiết lập các siêu tham số cấu trúc tương đồng (`max_depth=5`, `min_samples_split=20`, `min_samples_leaf=10`).
* **Kết quả thực nghiệm:** Đạt các chỉ số đánh giá **trùng khớp tuyệt đối 100% từng chữ số thập phân** với bản code chay, chứng minh logic thuật toán tự cài đặt hoàn toàn chuẩn xác theo giải thuật Breiman gốc.

---

# 5. ĐÁNH GIÁ HIỆU NĂNG MÔ HÌNH (EVALUATION)

## 5.1. Chiến lược chọn lựa độ đo

1. **RMSLE (Root Mean Squared Logarithmic Error):** Đo trực tiếp bằng sai số RMSE trên thang đo log. Đây là metric phản ánh sai số tỷ lệ phần trăm giữa các quy mô nhà hàng khác nhau một cách công bằng nhất.
2. **MAE & RMSE trên số khách thực tế:**
   * Nghịch đảo giá trị dự đoán về thang đo gốc: $\hat{y} = \exp(\hat{z}) - 1$.
   * Kẹp chặn giá trị không âm: $\hat{y} = \max(0, \hat{y})$.
   * Tính toán sai số trung bình tuyệt đối (MAE) và căn bậc hai sai số bình phương trung bình (RMSE).

## 5.2. Kết quả thực nghiệm tiêu biểu (trên mô hình CART)

* **RMSLE:** $\approx 0.7950$ (mức baseline tốt đối với cây đơn lẻ không dùng đặc trưng độ trễ lag).
* **RMSE:** $\approx 18.19$ khách.
* **MAE:** $\approx 12.79$ khách (trung bình sai lệch chỉ rơi vào khoảng $12 \sim 13$ khách/ngày).

### Bảng đối chiếu chi tiết 4 mô hình thực nghiệm:

| Mô hình | Phiên bản | RMSLE | RMSE (khách) | MAE (khách) |
| :--- | :--- | :---: | :---: | :---: |
| **Linear Regression** | Tự cài đặt (Ridge Scratch) | 0.7956 | 18.43 | 12.83 |
| **Linear Regression** | Thư viện (Scikit-Learn Ridge) | 0.7956 | 18.43 | 12.83 |
| **CART Regressor** | Tự cài đặt (CART Scratch) | **0.7950** | **18.19** | **12.79** |
| **CART Regressor** | Thư viện (Scikit-Learn CART) | **0.7950** | **18.19** | **12.79** |