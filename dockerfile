FROM python:3.11-slim

# JDK 및 빌드 도구 설치
RUN apt-get update && apt-get install -y \
    default-jdk \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# JAVA_HOME 자동 설정
RUN echo "export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))" >> /etc/profile
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="$JAVA_HOME/bin:$PATH"

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 복사
COPY . .

# Render가 주입하는 PORT 환경변수 사용
ENV PORT=8501
EXPOSE 8501

CMD streamlit run app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false
