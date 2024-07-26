# Base image olarak Python 3.9 kullanılıyor
FROM python:3.9-slim

# Gerekli sistem paketlerini yükleyin
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc

# Çalışma dizinini belirleme
WORKDIR /app

# Gerekli dosyaları çalışma dizinine kopyalama
COPY requirements.txt requirements.txt
COPY . .

# Gerekli Python paketlerini yükleme
RUN pip install --no-cache-dir -r requirements.txt

# Uygulamayı çalıştırma
CMD ["python", "app.py"]
