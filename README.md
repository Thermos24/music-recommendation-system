# 🎵 Music Recommendation System

## 📖 Giới thiệu dự án
Dự án **Music Recommendation System** được xây dựng nhằm mục đích khám phá, phân tích các thuộc tính đặc trưng của âm nhạc và ứng dụng các thuật toán Machine Learning để xây dựng hệ thống gợi ý bài hát thông minh. Dự án tập trung vào việc xử lý tập dữ liệu, phân tích trực quan hóa (EDA) và mô hình hóa dữ liệu nhằm đưa ra những đề xuất âm nhạc tối ưu nhất.

## 📁 Cấu trúc thư mục & Dữ liệu
Dưới đây là các tệp tin và thành phần chính trong kho lưu trữ này:
- `tcc_ceds_music.csv`: Tập dữ liệu gốc chứa thông tin chi tiết về các bài hát (các thuộc tính âm nhạc, thể loại, từ khóa...).
- **Hệ thống biểu đồ trực quan hóa (EDA):**
  - `heatmap.png`: Ma trận hiển thị mối tương quan giữa các đặc trưng âm nhạc.
  - `histogram.png`, `histo2.png`, `histo3.png`: Biểu đồ phân phối tần suất của các thuộc tính trong tập dữ liệu.
  - `boxplot3.png`: Biểu đồ hộp dùng để phát hiện và phân tích các giá trị ngoại lai (outliers).
  - `top30.png`: Biểu đồ trực quan hóa top các bài hát hoặc nghệ sĩ phổ biến.
  - `topic_gen.png`: Biểu đồ phân tích và phân loại các xu hướng chủ đề/thể loại âm nhạc.
  - `testlove.png`: Biểu đồ thử nghiệm và đánh giá phân tích đặc biệt trong dự án.

## 🛠️ Công nghệ sử dụng
Dự án được triển khai bằng ngôn ngữ **Python** cùng với các thư viện xử lý và phân tích dữ liệu chuyên sâu:
- **Quản lý & Xử lý dữ liệu:** `pandas`, `numpy`
- **Học máy (Machine Learning):** `scikit-learn`
- **Trực quan hóa dữ liệu:** `matplotlib`, `seaborn`

## 🚀 Hướng dẫn cài đặt và sử dụng
Để khởi chạy và kiểm tra dự án này trên máy tính cá nhân, bạn chỉ cần làm theo các bước đơn giản sau:

1. **Tải mã nguồn về máy:**
   Mở Terminal và chạy lệnh sau để clone repository:

       git clone https://github.com/Thermos24/music-recommendation-system.git
       cd music-recommendation-system

2. **Cài đặt môi trường:**
   Tại thư mục dự án, chạy lệnh sau để cài đặt các thư viện cần thiết:

       pip install pandas numpy scikit-learn matplotlib seaborn

3. **Khởi chạy dự án:**
   Sau khi cài xong, mở tệp mã nguồn (Python `.py`) để tiến hành huấn luyện mô hình và xem kết quả.
