# bass-sound-to-muse-score (`bass-mcp`)

로컬에서 실행되는 **MCP (Model Context Protocol) 서버 번들**입니다.  
유튜브 링크나 로컬 음원 파일을 입력하면 **베이스 트랙 분리 → MIDI/음표 채보 → 코드 추정 → 4현/5현 베이스 TAB + 오선 악보(MusicXML / MIDI / MuseScore `.mscz`)** 로 자동 변환합니다.

별도의 웹 서버나 유료 상용 API 없이, Claude Desktop / Claude Code / Cursor / Gemini CLI 등 MCP를 지원하는 모든 AI 클라이언트에 붙여 사용할 수 있습니다.

---

## 주요 기능 & 아키텍처

1. **보안 강화 입력 처리 (`security.py`, `downloader.py`)**
   - YouTube URL 엄격 검증 (`https://` + 공식 도메인 화이트리스트 + 11자 Video ID 정규화) → SSRF, 파일 URL(`file://`), 명령 플래그 인젝션 차단
   - 로컬 경로 검증 (symlink 차단, 경로 탈출 방지, Windows 예약 디바이스 이름 `CON`/`NUL`/`COM1` 등 차단, 200 MB 제한)
   - `shell=False` 및 최소 환경변수 allowlist로 서브프로세스 실행 (환경변수의 API 키/토큰 유출 방지)
2. **베이스 스템 분리 (`separator.py`)**
   - [`facebookresearch/demucs`](https://github.com/facebookresearch/demucs) (`htdemucs --two-stems=bass`) 우선 지원
   - 미설치 시 내장 `scipy` 4차 Butterworth 대역통과 필터(`30 Hz ~ 350 Hz`) 자동 폴백
3. **베이스 음정/리듬 채보 (`transcriber.py`)**
   - [`spotify/basic-pitch`](https://github.com/spotify/basic-pitch) 우선 지원 (피치 벤드 포함)
   - `librosa.pyin` 및 내장 순수 NumPy/SciPy YIN 자기상관 피치 트래커 자동 폴백
   - 4현(`E1=28`~`G4=67`) / 5현(`B0=23`~`G4=67`) 베이스 음역 필터링
4. **베이스 TAB · 코드 · 악보 생성 (`score_builder.py`)**
   - 동적 계획법(DP) 기반 지판 포지션 최적화(손 이동 최소화)
   - 마디별 강박 루트 + 3도/7도 기반 코드 심볼 추정
   - **2-Staff MusicXML 4.0** (베이스 오선보 + 4현/5현 TAB + 코드 심볼), **표준 MIDI (`.mid`)**, **ASCII Bass TAB** 생성
   - 로컬 MuseScore 4 CLI 감지 시 `.mscz` 자동 변환
5. **실시간 MuseScore 4 제어 (`musescore_bridge.py`, `plugins/musescore-mcp-websocket.qml`)**
   - [`ghchen99/mcp-musescore`](https://github.com/ghchen99/mcp-musescore) WebSocket QML 브리지 패턴 적용 (`127.0.0.1` 루프백 전용 제한)

---

## 설치 방법

### 1. 기본 설치 (경량/즉시 실행 — 내장 필터 + YIN + MIDI/MusicXML 포함)
```powershell
python -m venv .venv
.venv\Scripts\pip install -e .
```

### 2. 고품질 ML 엔진 포함 설치 (Demucs + Spotify Basic Pitch)
> 참고: `ffmpeg`가 시스템 `PATH`에 설치되어 있어야 유튜브/MP3/M4A 변환이 가능합니다.
```powershell
.venv\Scripts\pip install -e ".[ml]"
```

### 3. (선택) MuseScore 4 실시간 연동 플러그인
1. [`plugins/musescore-mcp-websocket.qml`](plugins/musescore-mcp-websocket.qml) 파일을 `%USERPROFILE%\Documents\MuseScore4\Plugins\` 폴더에 복사합니다.
2. MuseScore 4 → **Plugins → Plugin Manager**에서 활성화한 뒤 실행합니다 (`ws://127.0.0.1:8765`).

---

## MCP 클라이언트 연동 설정 (`mcp_config.example.json`)

Claude Desktop, Cursor, Claude Code, Gemini CLI 등의 MCP 설정 파일에 아래 항목을 추가하세요.

```json
{
  "mcpServers": {
    "bass-sound-to-musescore": {
      "command": "C:\\dev\\bass-sound-to-muse-score\\.venv\\Scripts\\python.exe",
      "args": ["-m", "bass_mcp"],
      "env": {
        "BASS_MCP_OUTPUT_DIR": "C:\\dev\\bass-sound-to-muse-score\\output"
      }
    }
  }
}
```

(선택 보안 옵션: `BASS_MCP_ALLOWED_DIRS` 환경변수에 `;`로 구분된 허용 폴더를 지정하면 해당 폴더 밖의 로컬 음원 접근을 차단합니다.)

---

## 제공 MCP Tools

| Tool | 설명 |
|---|---|
| `transcribe_bass_pipeline` | 유튜브 URL 또는 로컬 음원 → 베이스 분리 → 채보 → 코드/TAB/MusicXML/MIDI/MSCZ 원스톱 실행 |
| `separate_bass_track` | 유튜브 URL 또는 음원에서 베이스 트랙 WAV만 분리 |
| `audio_to_bass_score` | 이미 분리된 베이스 오디오 파일을 악보(`.musicxml`, `.mid`, ASCII TAB)로 변환 |
| `musescore_live_push` | 생성된 노트 JSON 또는 MusicXML을 실행 중인 MuseScore 4로 전송 (`127.0.0.1:8765`) |
| `check_environment_status` | `ffmpeg`, `yt-dlp`, `demucs`, `basic-pitch`, MuseScore CLI 설치 상태 및 라이선스 정보 확인 |

---

## 오픈소스 라이선스 및 출처 표기 (Attributions)

자세한 전문과 고지사항은 [LICENSE](LICENSE), [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)를 참고하세요.

- **Spotify Basic Pitch** (`spotify/basic-pitch`) — Apache License 2.0
- **Meta / Facebook Research Demucs** (`facebookresearch/demucs`) — MIT License (사전학습 모델 가중치는 별도 데이터셋 약관 확인 필요)
- **ghchen99/mcp-musescore** (`ghchen99/mcp-musescore`) — MIT License
- **sandraschi/stems-mcp** (`sandraschi/stems-mcp`) — MIT License
- **pretty_midi** (`craffel/pretty-midi`) — MIT License
- **yt-dlp** (`yt-dlp/yt-dlp`) — The Unlicense
- **MCP Python SDK** (`modelcontextprotocol/python-sdk`) — MIT License
