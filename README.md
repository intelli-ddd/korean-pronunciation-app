# 🇰🇷 Korean Pronunciation Practice (한국어 발음 교정)

베트남어 UI로 제공되는 한국어 발음 교정 웹앱입니다.

## 기능
- 🎙️ 문장 선택 → 녹음 → 발음 분석
- 📊 실시간 피치(억양) 가이드 그래프
- 🔊 표준 발음 듣기 (여성/남성 선택)
- 🎯 종합 점수 + 상세 피드백
- 🇻🇳 베트남어 인터페이스

## 기술 스택
- **Backend**: FastAPI + Parselmouth (Praat) + edge-tts
- **Frontend**: Vanilla HTML/JS + Web Audio API + YIN pitch detection
- **TTS**: Microsoft Edge Neural TTS (한국어)

## 로컬 실행
```bash
pip install -r requirements.txt
python app.py
```

## Docker
```bash
docker build -t korean-pronunciation .
docker run -p 8080:8080 korean-pronunciation
```

## 배포
- **Cloud Run**: `gcloud run deploy` (Dockerfile 기반)
- **Vercel**: 프론트엔드 정적 배포
