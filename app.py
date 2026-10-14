    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = float(np.sqrt(mean_squared_error(y_test_real, test_pred)))
    mae = float(mean_absolute_error(y_test_real, test_pred))
    r2 = float(r2_score(y_test_real, test_pred)) if len(y_test_real) > 1 else float("nan")

    # Dự báo đệ quy từ 60 phiên gần nhất có sẵn
    last_60_days = scaler.transform(data)[-60:].reshape(1, 60, 1)
    future_scaled = []

    for _ in range(forecast_days):
        next_pred = model.predict(last_60_days, verbose=0)
        future_scaled.append(float(next_pred[0, 0]))
        last_60_days = np.concatenate(
            [last_60_days[:, 1:, :], next_pred.reshape(1, 1, 1)],
            axis=1
        )

    future_predictions = scaler.inverse_transform(
        np.array(future_scaled).reshape(-1, 1)
    ).flatten()

# -----------------------------
# Kết quả
# -----------------------------
st.success("Đã hoàn tất huấn luyện và dự báo.")

st.subheader("1. Đánh giá mô hình trên tập kiểm tra")
m1, m2, m3 = st.columns(3)
m1.metric("RMSE", f"{rmse:,.2f}")
m2.metric("MAE", f"{mae:,.2f}")
m3.metric("R²", f"{r2:.4f}" if np.isfinite(r2) else "Không xác định")

st.caption(
    "RMSE và MAE càng thấp thường càng tốt. R² gần 1 cho thấy mô hình giải thích "
    "được nhiều biến thiên trong tập kiểm tra; R² có thể âm nếu dự báo kém hơn "
    "mốc dự báo bằng giá trị trung bình."
)

st.subheader("2. Loss trong quá trình huấn luyện")
fig_loss, ax_loss = plt.subplots(figsize=(10, 4))
ax_loss.plot(history.history["loss"], label="Train Loss")
ax_loss.plot(history.history["val_loss"], label="Validation Loss")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Mean Squared Error")
ax_loss.set_title("Train Loss vs Validation Loss")
ax_loss.legend()
ax_loss.grid(alpha=0.25)
st.pyplot(fig_loss)
plt.close(fig_loss)

st.subheader("3. Giá thực tế và giá dự báo trên tập kiểm tra")
test_dates = stock_df.index[train_size:]
comparison = pd.DataFrame(
    {
        "Giá thực tế": y_test_real.flatten(),
        "Giá dự báo": test_pred.flatten()
    },
    index=test_dates
)
st.line_chart(comparison, use_container_width=True)

with st.expander("Xem bảng so sánh tập kiểm tra"):
    st.dataframe(comparison, use_container_width=True)

st.subheader(f"4. Dự báo {forecast_days} phiên giao dịch tiếp theo")

# Tạo ngày làm việc dự kiến; đây không phải lịch giao dịch chính thức của sàn
future_dates = pd.bdate_range(
    start=stock_df.index[-1] + pd.Timedelta(days=1),
    periods=forecast_days
)

future_df = pd.DataFrame(
    {
        "Ngày dự kiến": future_dates,
        "Giá dự báo": future_predictions
    }
)
future_df["Ngày dự kiến"] = future_df["Ngày dự kiến"].dt.strftime("%Y-%m-%d")

st.dataframe(
    future_df.style.format({"Giá dự báo": "{:,.2f}"}),
    use_container_width=True,
    hide_index=True
)

fig_future, ax_future = plt.subplots(figsize=(12, 5))
recent_count = min(90, len(stock_df))
recent_history = stock_df.iloc[-recent_count:]

ax_future.plot(
    recent_history.index,
    recent_history["Close"],
    label="Giá đóng cửa lịch sử"
)
ax_future.plot(
    future_dates,
    future_predictions,
    marker="o",
    linestyle="--",
    label="Giá dự báo"
)
ax_future.set_title(f"{ticker}: lịch sử gần đây và dự báo tương lai")
ax_future.set_xlabel("Ngày")
ax_future.set_ylabel("Giá đóng cửa")
ax_future.legend()
ax_future.grid(alpha=0.25)
fig_future.autofmt_xdate()
st.pyplot(fig_future)
plt.close(fig_future)

csv = future_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "⬇️ Tải bảng dự báo CSV",
    data=csv,
    file_name=f"{ticker.lower().replace('.', '_')}_forecast.csv",
    mime="text/csv"
)

st.warning(
    "Lưu ý: Giá tương lai được dự báo đệ quy từ chính các dự báo trước đó, "
    "nên sai số có thể tích lũy theo số phiên. Ngày dự kiến chỉ là ngày làm việc "
    "ước tính, không loại trừ ngày nghỉ lễ của thị trường. Mô hình không đảm bảo "
    "dự báo chính xác và không nên dùng làm căn cứ đầu tư duy nhất."
)
