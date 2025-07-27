FROM python:3.9-slim

# Tránh các prompt tương tác trong quá trình cài đặt
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Sao chép requirements trước để tận dụng cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn ứng dụng
COPY . .

# Tạo các thư mục cần thiết
RUN mkdir -p uploads static/img static/css static/js static/lib static/results data

# Expose port
EXPOSE 5000

# Kiểm tra trạng thái ứng dụng
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/ || exit 1

# Khởi động ứng dụng
CMD ["python", "app.py"]