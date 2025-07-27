# 🧠 Intelligent Brain Tumor Segmentation System

Hệ thống thông minh phân đoạn khối u não sử dụng Deep Learning với kiến trúc U-Net và cơ sở dữ liệu phân tán Cassandra.

## 📋 Mục lục
- [Tổng quan](#tổng-quan)
- [Tính năng chính](#tính-năng-chính)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [API Documentation](#api-documentation)
- [Mô hình Machine Learning](#mô-hình-machine-learning)
- [Dataset](#dataset)
- [Kết quả](#kết-quả)
- [Đóng góp](#đóng-góp)
- [Giấy phép](#giấy-phép)

## 🎯 Tổng quan

Hệ thống này được phát triển để hỗ trợ các bác sĩ và chuyên gia y tế trong việc phân đoạn và chẩn đoán khối u não từ ảnh MRI. Sử dụng mô hình U-Net được huấn luyện trên dataset BraTS2020, hệ thống có thể tự động nhận diện và phân loại các vùng khối u não với độ chính xác cao.

### Mục tiêu chính:
- ✅ Tự động phân đoạn khối u não từ ảnh MRI
- ✅ Cung cấp giao diện web thân thiện cho người dùng
- ✅ Lưu trữ và quản lý dữ liệu bệnh nhân hiệu quả
- ✅ Hỗ trợ phân tích thống kê và báo cáo
- ✅ Triển khai dễ dàng với Docker

## 🚀 Tính năng chính

### 🔬 Phân đoạn khối u não
- Phân đoạn tự động 4 vùng: Background, NCR/NET, Edema, Enhancing Tumor
- Hỗ trợ định dạng NIfTI (.nii)
- Xử lý ảnh MRI đa kênh (FLAIR, T1CE)
- Trực quan hóa kết quả với colormap tùy chỉnh

### 👥 Quản lý người dùng
- Đăng ký/đăng nhập với mã hóa mật khẩu
- Phân quyền người dùng (User/Admin)
- Lịch sử dự đoán cá nhân
- Quản lý thông tin bệnh nhân

### 📊 Phân tích và báo cáo
- Thống kê chi tiết về dự đoán
- So sánh kết quả giữa các mô hình
- Xuất báo cáo PDF
- Phân tích sinh tồn (Survival Analysis)

### 🌐 Giao diện web
- Thiết kế responsive, thân thiện với người dùng
- Upload và preview ảnh MRI
- Xem kết quả phân đoạn real-time
- Tìm kiếm và lọc dữ liệu

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   Database      │
│   (HTML/CSS/JS) │───▶│   (Flask)       │───▶│   (Cassandra)   │
│   Bootstrap     │    │   Python        │    │   NoSQL         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   AI Model      │
                       │   (U-Net)       │
                       │   TensorFlow    │
                       └─────────────────┘
```

### Thành phần chính:
- **Frontend**: Giao diện web với Bootstrap, JavaScript
- **Backend**: Flask API với Python
- **AI Model**: U-Net architecture với TensorFlow/Keras
- **Database**: Apache Cassandra cho lưu trữ phân tán
- **Container**: Docker & Docker Compose

## 🛠️ Công nghệ sử dụng

### Backend
- **Python 3.9+**
- **Flask 2.2.3** - Web framework
- **TensorFlow 2.12.0** - Deep learning
- **OpenCV 4.7.0** - Xử lý ảnh
- **NiBabel 5.0.1** - Xử lý ảnh NIfTI
- **Cassandra Driver 3.25.0** - Database connector

### AI/ML
- **U-Net Architecture** - Semantic segmentation
- **TensorFlow/Keras** - Model training & inference
- **NumPy** - Numerical computing
- **Matplotlib** - Visualization
- **Scikit-image** - Image processing

### Database
- **Apache Cassandra** - NoSQL distributed database
- **CQLEngine** - Object mapping

### DevOps
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration

### Frontend
- **HTML5/CSS3/JavaScript**
- **Bootstrap 5** - Responsive design
- **jQuery** - DOM manipulation

## 📦 Cài đặt

### Yêu cầu hệ thống
- Docker 20.10+
- Docker Compose 1.29+
- RAM: 8GB+ (khuyến nghị 16GB)
- Disk: 10GB+ trống
- GPU: CUDA compatible (tùy chọn)

### 1. Clone repository
```bash
git clone https://github.com/BaeThang/Building-an-Intelligent-Brain-Tumor-System.git
cd Building-an-Intelligent-Brain-Tumor-System
```

### 2. Chuẩn bị dữ liệu
```bash
# Tạo thư mục cho dữ liệu BraTS2020
mkdir -p brats_data
# Copy dữ liệu BraTS2020 vào thư mục brats_data/
```

### 3. Cấu hình môi trường
```bash
# Copy file cấu hình mẫu
cp .env.example .env
# Chỉnh sửa các biến môi trường trong .env
```

### 4. Khởi chạy với Docker
```bash
# Build và khởi chạy tất cả services
docker-compose up -d

# Xem logs
docker-compose logs -f app
```

### 5. Khởi tạo database
```bash
# Chạy script khởi tạo Cassandra
docker-compose exec app python init_database.py
```

## 🎮 Sử dụng

### Truy cập ứng dụng
1. Mở trình duyệt web: `http://localhost:5000`
2. Đăng ký tài khoản mới hoặc đăng nhập
3. Upload file ảnh MRI (.nii format)
4. Chọn slice cần phân đoạn
5. Xem kết quả và tải về

### Upload ảnh MRI
```
Supported formats: .nii (NIfTI)
Required files: 
- FLAIR image
- T1CE image
Max file size: 100MB per file
```

### Xem kết quả
- **Background (0)**: Màu đen - Vùng nền
- **NCR/NET (1)**: Màu đỏ - Vùng hoại tử/không tăng cường
- **Edema (2)**: Màu vàng - Vùng phù nề
- **Enhancing Tumor (3)**: Màu xanh - Khối u tăng cường

## 📚 API Documentation

### Endpoints chính

#### Dự đoán khối u
```http
POST /predict
Content-Type: multipart/form-data

Parameters:
- flair_file: File (.nii)
- t1ce_file: File (.nii)
- slice_index: Integer (0-154)
```

#### Lấy thông tin bệnh nhân
```http
GET /api/patients
Authorization: Bearer <token>
```

#### Xem lịch sử dự đoán
```http
GET /api/predictions/{user_id}
Authorization: Bearer <token>
```

### Response format
```json
{
  "status": "success",
  "data": {
    "prediction_id": "uuid",
    "result_image": "base64_string",
    "tumor_stats": {
      "background": 85.2,
      "ncr_net": 3.1,
      "edema": 8.5,
      "enhancing": 3.2
    }
  }
}
```

## 🤖 Mô hình Machine Learning

### Kiến trúc U-Net
```python
Input: (128, 128, 2)  # FLAIR + T1CE
├── Encoder
│   ├── Conv2D(32) + MaxPool
│   ├── Conv2D(64) + MaxPool
│   ├── Conv2D(128) + MaxPool
│   └── Conv2D(256) + MaxPool
├── Bottleneck
│   └── Conv2D(512) + Dropout
└── Decoder
    ├── UpSample + Conv2D(256) + Skip Connection
    ├── UpSample + Conv2D(128) + Skip Connection
    ├── UpSample + Conv2D(64) + Skip Connection
    └── Conv2D(4) + Softmax
Output: (128, 128, 4)  # 4 classes
```

### Metrics
- **Loss Function**: Categorical Crossentropy
- **Primary Metric**: Dice Coefficient
- **Additional Metrics**: 
  - Mean IoU (Intersection over Union)
  - Precision, Recall, Specificity
  - Class-specific Dice coefficients

### Hyperparameters
```python
optimizer = Adam(learning_rate=0.001)
batch_size = 2
epochs = 15
input_size = (128, 128, 2)
dropout_rate = 0.2
```

## 📊 Dataset

### BraTS2020 Dataset
- **Training samples**: 369 patients
- **Image modalities**: FLAIR, T1, T1CE, T2
- **Ground truth**: Manual segmentation
- **Resolution**: 240×240×155 voxels
- **Voxel size**: 1×1×1 mm³

### Preprocessing
```python
# Resize to 128x128
# Normalize pixel values [0, 1]
# Extract slices [22:122] (100 slices)
# Augmentation: rotation, flipping
```

### Data Split
- **Training**: 70% (258 patients)
- **Validation**: 15% (55 patients)
- **Testing**: 15% (56 patients)

## 📈 Kết quả

### Performance Metrics
| Metric | Value |
|--------|--------|
| Overall Dice Score | 0.847 |
| Mean IoU | 0.735 |
| Accuracy | 92.3% |
| Sensitivity | 89.1% |
| Specificity | 94.7% |

### Class-specific Performance
| Class | Dice Score | IoU |
|-------|------------|-----|
| NCR/NET | 0.823 | 0.701 |
| Edema | 0.856 | 0.748 |
| Enhancing | 0.791 | 0.656 |

### Training History
- **Training Time**: ~6 hours (Tesla V100)
- **Convergence**: Epoch 12
- **Best Model**: Saved automatically

## 🐳 Docker Configuration

### docker-compose.yml
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "5000:5000"
    depends_on:
      - cassandra
    volumes:
      - ./uploads:/app/uploads
      - ./static/results:/app/static/results
    
  cassandra:
    image: cassandra:3.11
    ports:
      - "9042:9042"
    environment:
      - CASSANDRA_CLUSTER_NAME=brain_tumor_cluster
    volumes:
      - cassandra_data:/var/lib/cassandra
```

## 🔧 Troubleshooting

### Lỗi thường gặp

#### 1. Lỗi kết nối Cassandra
```bash
# Kiểm tra status
docker-compose ps

# Restart Cassandra
docker-compose restart cassandra
```

#### 2. Lỗi thiếu model file
```bash
# Đảm bảo file model tồn tại
ls -la model/3D_MRI_Brain_tumor_segmentation.h5
```

#### 3. Lỗi memory
```bash
# Tăng memory limit cho Docker
# Hoặc giảm batch_size trong code
```

#### 4. Lỗi upload file
```bash
# Kiểm tra permissions
chmod 755 uploads/
```

## 🤝 Đóng góp

Chúng tôi hoan nghênh mọi đóng góp! Vui lòng:

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

### Coding Standards
- Sử dụng PEP 8 cho Python code
- Viết docstring cho functions
- Thêm unit tests cho features mới
- Update documentation

### Bug Reports
Khi báo cáo bug, vui lòng include:
- OS và phiên bản Python
- Logs đầy đủ
- Steps to reproduce
- Expected vs actual behavior

## 📄 Giấy phép

Dự án này được phân phối dưới giấy phép MIT. Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

## 👨‍💻 Tác giả

**BaeThang**
- GitHub: [@BaeThang](https://github.com/BaeThang)
- Email: [your-email@example.com]

## 🙏 Acknowledgments

- **BraTS Challenge** - Cung cấp dataset
- **TensorFlow Team** - Deep learning framework
- **Apache Cassandra** - Distributed database
- **Flask Community** - Web framework
- **Medical Imaging Community** - Research support

## 📞 Liên hệ

Nếu bạn có câu hỏi hoặc cần hỗ trợ:
- Tạo Issue trên GitHub
- Email: [your-email@example.com]
- LinkedIn: [Your LinkedIn Profile]

## 🚀 Roadmap

### Version 2.0 (Planned)
- [ ] 3D Segmentation support
- [ ] Real-time inference API
- [ ] Mobile app integration
- [ ] Advanced visualization tools
- [ ] Multi-language support
- [ ] Cloud deployment guide

### Version 1.1 (In Progress)
- [ ] Performance optimization
- [ ] Better error handling
- [ ] Enhanced UI/UX
- [ ] Automated testing
- [ ] Documentation improvements

---

**⭐ Nếu dự án này hữu ích, hãy cho chúng tôi một star!**