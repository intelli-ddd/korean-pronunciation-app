FROM python:3.12-slim

# ffmpeg 설치
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 복사
COPY app.py .
COPY templates/ templates/
COPY static/ static/

# reference_audio 디렉토리 생성
RUN mkdir -p reference_audio

# Cloud Run은 PORT 환경변수 사용
ENV PORT=8080

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]
