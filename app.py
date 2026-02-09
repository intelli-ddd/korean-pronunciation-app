"""
한국어 발음 교정 앱 - 프로토타입
무료 오픈소스만 사용:
- Web Speech API (브라우저 내장, 무료)
- Parselmouth/Praat (피치 분석, 오픈소스)
- gTTS (표준 발음 생성, 무료)
- mlx-whisper (로컬 음성인식, 무료)
"""

import os
import io
import json
import tempfile
import wave
import struct
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import numpy as np
import parselmouth
from parselmouth.praat import call
import asyncio
import edge_tts

app = FastAPI(title="한국어 발음 교정")

BASE_DIR = Path(__file__).parent
REFERENCE_DIR = BASE_DIR / "reference_audio"
REFERENCE_DIR.mkdir(exist_ok=True)

# ── 표준 발음 생성 (edge-tts) ──────────────────────

VOICE_MAP = {
    "female": "ko-KR-SunHiNeural",
    "male": "ko-KR-InJoonNeural",
}

async def get_reference_audio(text: str, voice: str = "female") -> Path:
    """표준 발음 오디오 파일을 생성하거나 캐시에서 반환"""
    safe_name = text.replace(" ", "_")[:50]
    voice_key = voice if voice in VOICE_MAP else "female"
    ref_path = REFERENCE_DIR / f"{safe_name}_{voice_key}.mp3"
    wav_path = REFERENCE_DIR / f"{safe_name}_{voice_key}.wav"
    
    if not wav_path.exists():
        tts_voice = VOICE_MAP[voice_key]
        communicate = edge_tts.Communicate(text, tts_voice, rate="-30%")  # 30% 느리게
        await communicate.save(str(ref_path))
        # mp3 → wav 변환 (parselmouth용)
        os.system(f'ffmpeg -y -i "{ref_path}" -ar 16000 -ac 1 "{wav_path}" 2>/dev/null')
    
    return wav_path


# ── 무음 제거 (트리밍) ─────────────────────────────

def trim_silence(audio_path: str, silence_threshold_db: float = -35.0, min_silence_dur: float = 0.05) -> str:
    """오디오 앞뒤 무음 구간을 제거하고 트리밍된 파일 경로 반환"""
    snd = parselmouth.Sound(audio_path)
    intensity = snd.to_intensity(time_step=0.01)
    
    times = intensity.xs()
    values = [intensity.get_value(t) for t in times]
    
    # 최대 강도 대비 임계값 계산
    max_int = max(v for v in values if not np.isnan(v)) if values else 0
    if max_int == 0:
        return audio_path
    
    threshold = max_int + silence_threshold_db  # dB 상대값
    
    # 음성이 시작되는 첫 지점
    start_time = 0
    for t, v in zip(times, values):
        if not np.isnan(v) and v >= threshold:
            start_time = max(0, t - 0.02)  # 약간 여유
            break
    
    # 음성이 끝나는 마지막 지점
    end_time = float(snd.duration)
    for t, v in reversed(list(zip(times, values))):
        if not np.isnan(v) and v >= threshold:
            end_time = min(float(snd.duration), t + 0.02)
            break
    
    # 트리밍
    if end_time - start_time < 0.1:
        return audio_path  # 너무 짧으면 원본 반환
    
    trimmed = snd.extract_part(start_time, end_time)
    trimmed_path = audio_path.replace('.wav', '_trimmed.wav')
    trimmed.save(trimmed_path, "WAV")
    return trimmed_path


# ── 피치 분석 ──────────────────────────────────────

def extract_pitch_data(audio_path: str, time_step=0.01, do_trim=True):
    """오디오에서 피치(F0) 데이터 추출 (무음 트리밍 포함)"""
    # 무음 구간 제거
    if do_trim:
        trimmed_path = trim_silence(audio_path)
    else:
        trimmed_path = audio_path
    
    snd = parselmouth.Sound(trimmed_path)
    pitch = snd.to_pitch(time_step=time_step)
    
    times = pitch.xs()
    frequencies = []
    for t in times:
        f = pitch.get_value_at_time(t)
        frequencies.append(float(f) if not np.isnan(f) else 0)
    
    # 강도(intensity) 추출
    intensity = snd.to_intensity(time_step=time_step)
    intensity_values = []
    for t in times:
        val = intensity.get_value(t)
        intensity_values.append(float(val) if not np.isnan(val) else 0)
    
    result = {
        "times": [round(t, 3) for t in times],
        "frequencies": [round(f, 1) for f in frequencies],
        "intensity": [round(i, 1) for i in intensity_values],
        "duration": round(float(snd.duration), 3),
        "mean_pitch": round(float(call(pitch, "Get mean", 0, 0, "Hertz")), 1),
        "std_pitch": round(float(call(pitch, "Get standard deviation", 0, 0, "Hertz")), 1),
    }
    
    # 트리밍된 임시 파일 정리
    if trimmed_path != audio_path and os.path.exists(trimmed_path):
        os.unlink(trimmed_path)
    
    return result


def compare_pitch(user_data: dict, ref_data: dict) -> dict:
    """사용자 발음과 표준 발음의 피치 비교 분석"""
    # 시간축 정규화 (0~1)
    def normalize_time(data):
        dur = data["duration"]
        if dur == 0:
            return data["times"], data["frequencies"]
        norm_times = [t / dur for t in data["times"]]
        return norm_times, data["frequencies"]
    
    user_times, user_freqs = normalize_time(user_data)
    ref_times, ref_freqs = normalize_time(ref_data)
    
    # 유효한 피치만 비교
    user_valid = [(t, f) for t, f in zip(user_times, user_freqs) if f > 0]
    ref_valid = [(t, f) for t, f in zip(ref_times, ref_freqs) if f > 0]
    
    if not user_valid or not ref_valid:
        return {"score": 0, "feedback": "음성이 감지되지 않았습니다."}
    
    # 피치 범위 비교
    user_range = max(f for _, f in user_valid) - min(f for _, f in user_valid)
    ref_range = max(f for _, f in ref_valid) - min(f for _, f in ref_valid)
    
    # 피치 패턴 유사도 (상관계수 기반)
    # 보간하여 같은 수의 포인트로 맞춤
    n_points = 50
    user_interp = np.interp(
        np.linspace(0, 1, n_points),
        [t for t, _ in user_valid],
        [f for _, f in user_valid]
    )
    ref_interp = np.interp(
        np.linspace(0, 1, n_points),
        [t for t, _ in ref_valid],
        [f for _, f in ref_valid]
    )
    
    # 정규화 후 상관계수
    user_norm = (user_interp - np.mean(user_interp)) / (np.std(user_interp) + 1e-6)
    ref_norm = (ref_interp - np.mean(ref_interp)) / (np.std(ref_interp) + 1e-6)
    
    correlation = float(np.corrcoef(user_norm, ref_norm)[0, 1])
    if np.isnan(correlation):
        correlation = 0
    
    # 점수 산출 (0~100)
    pitch_score = max(0, min(100, int((correlation + 1) * 50)))
    
    # 피드백 생성
    feedbacks = []
    
    if user_data["mean_pitch"] > ref_data["mean_pitch"] * 1.3:
        feedbacks.append("전체적으로 음이 높습니다. 조금 낮은 톤으로 말해보세요.")
    elif user_data["mean_pitch"] < ref_data["mean_pitch"] * 0.7:
        feedbacks.append("전체적으로 음이 낮습니다. 조금 높은 톤으로 말해보세요.")
    
    if user_range < ref_range * 0.5:
        feedbacks.append("억양의 변화가 적습니다. 좀 더 자연스럽게 높낮이를 넣어보세요.")
    elif user_range > ref_range * 1.5:
        feedbacks.append("억양의 변화가 너무 큽니다. 좀 더 일정하게 말해보세요.")
    
    if pitch_score >= 80:
        feedbacks.insert(0, "👏 아주 좋습니다! 억양이 자연스럽습니다.")
    elif pitch_score >= 60:
        feedbacks.insert(0, "👍 괜찮습니다! 조금만 더 연습하면 완벽해요.")
    elif pitch_score >= 40:
        feedbacks.insert(0, "💪 표준 발음을 듣고 억양 패턴을 따라해보세요.")
    else:
        feedbacks.insert(0, "📖 표준 발음을 여러 번 듣고 천천히 따라해보세요.")
    
    return {
        "score": pitch_score,
        "correlation": round(correlation, 3),
        "feedback": feedbacks,
        "user_mean_pitch": user_data["mean_pitch"],
        "ref_mean_pitch": ref_data["mean_pitch"],
        "user_pitch_range": round(user_range, 1),
        "ref_pitch_range": round(ref_range, 1),
    }


# ── API 엔드포인트 ─────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.post("/api/analyze")
async def analyze_pronunciation(
    audio: UploadFile = File(...),
    text: str = Form(...),
    recognized_text: str = Form(""),
    voice: str = Form("female"),
):
    """사용자 음성을 분석하여 발음 피드백 제공"""
    
    # 1. 사용자 오디오 저장 + WAV 변환
    content = await audio.read()
    
    # 파일 확장자 결정 (업로드된 파일명 기반)
    fname = audio.filename or "recording.webm"
    if fname.endswith(".wav"):
        orig_suffix = ".wav"
    elif fname.endswith(".webm"):
        orig_suffix = ".webm"
    elif fname.endswith(".ogg"):
        orig_suffix = ".ogg"
    elif fname.endswith(".mp4") or fname.endswith(".m4a"):
        orig_suffix = ".m4a"
    else:
        orig_suffix = ".webm"  # 기본값
    
    # 원본 파일 저장
    with tempfile.NamedTemporaryFile(suffix=orig_suffix, delete=False) as tmp:
        tmp.write(content)
        raw_audio_path = tmp.name
    
    # WAV가 아니면 ffmpeg로 변환
    if orig_suffix != ".wav":
        wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_tmp.close()
        user_audio_path = wav_tmp.name
        ret = os.system(f'ffmpeg -y -i "{raw_audio_path}" -ar 16000 -ac 1 "{user_audio_path}" 2>/dev/null')
        os.unlink(raw_audio_path)
        if ret != 0:
            try:
                os.unlink(user_audio_path)
            except:
                pass
            return JSONResponse({"success": False, "error": "오디오 변환 실패. 다시 녹음해주세요."}, status_code=400)
    else:
        # WAV인 경우 유효성 검증 후 사용
        user_audio_path = raw_audio_path
        # WAV 헤더 확인
        if len(content) < 44 or content[:4] != b'RIFF':
            # WAV가 아니면 ffmpeg로 변환 시도
            wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav_tmp.close()
            converted_path = wav_tmp.name
            ret = os.system(f'ffmpeg -y -i "{raw_audio_path}" -ar 16000 -ac 1 "{converted_path}" 2>/dev/null')
            os.unlink(raw_audio_path)
            if ret != 0:
                try:
                    os.unlink(converted_path)
                except:
                    pass
                return JSONResponse({"success": False, "error": "오디오 형식 인식 실패"}, status_code=400)
            user_audio_path = converted_path
    
    try:
        # 2. 표준 발음 생성
        ref_audio_path = await get_reference_audio(text, voice=voice)
        
        # 3. 피치 분석
        user_pitch = extract_pitch_data(user_audio_path)
        ref_pitch = extract_pitch_data(str(ref_audio_path))
        
        # 4. 비교 분석
        comparison = compare_pitch(user_pitch, ref_pitch)
        
        # 5. 텍스트 비교 (Web Speech API 결과 활용)
        text_analysis = analyze_text_diff(text, recognized_text)
        
        return JSONResponse({
            "success": True,
            "user_pitch": user_pitch,
            "ref_pitch": ref_pitch,
            "comparison": comparison,
            "text_analysis": text_analysis,
        })
    
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)
    
    finally:
        os.unlink(user_audio_path)


def analyze_text_diff(expected: str, recognized: str) -> dict:
    """기대 텍스트와 인식된 텍스트 비교"""
    if not recognized:
        return {"match": False, "score": 0, "details": "음성 인식 결과가 없습니다."}
    
    # 공백 제거 후 비교
    exp_clean = expected.replace(" ", "")
    rec_clean = recognized.replace(" ", "")
    
    if exp_clean == rec_clean:
        return {
            "match": True,
            "score": 100,
            "expected": expected,
            "recognized": recognized,
            "details": "✅ 완벽합니다! 정확하게 발음했습니다.",
        }
    
    # 글자 단위 비교
    matches = 0
    char_results = []
    
    # 간단한 LCS 기반 매칭
    for i, char in enumerate(exp_clean):
        if i < len(rec_clean) and exp_clean[i] == rec_clean[i]:
            matches += 1
            char_results.append({"char": char, "correct": True})
        else:
            rec_char = rec_clean[i] if i < len(rec_clean) else "?"
            char_results.append({"char": char, "correct": False, "heard": rec_char})
    
    score = int(matches / len(exp_clean) * 100) if exp_clean else 0
    
    wrong_chars = [r for r in char_results if not r["correct"]]
    details = []
    if wrong_chars:
        for w in wrong_chars:
            heard = w.get("heard", "?")
            details.append(f"'{w['char']}' → '{heard}'로 들림")
    
    return {
        "match": False,
        "score": score,
        "expected": expected,
        "recognized": recognized,
        "char_results": char_results,
        "details": details if details else ["일부 발음이 다르게 인식되었습니다."],
    }


@app.get("/api/reference/{text}")
async def get_reference(text: str, voice: str = "female"):
    """표준 발음 오디오 반환"""
    wav_path = await get_reference_audio(text, voice=voice)
    mp3_path = wav_path.with_suffix(".mp3")
    if mp3_path.exists():
        return FileResponse(str(mp3_path), media_type="audio/mpeg")
    return FileResponse(str(wav_path), media_type="audio/wav")


@app.get("/api/reference_pitch/{text}")
async def get_reference_pitch(text: str, voice: str = "female"):
    """표준 발음의 피치 데이터 반환 (그래프 미리 그리기용)"""
    try:
        wav_path = await get_reference_audio(text, voice=voice)
        pitch_data = extract_pitch_data(str(wav_path))
        return JSONResponse({"success": True, "pitch": pitch_data})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# Static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        ssl_certfile="cert.pem",
        ssl_keyfile="key.pem",
    )
