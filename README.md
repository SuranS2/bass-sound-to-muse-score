# bass-sound-to-muse-score (`bass-mcp`)

> [!IMPORTANT]
> **이용 목적 및 저작권 책임 안내 (필독)**
> - **서비스 목적:** 본 도구는 개인 합주·연습 준비를 위한 **베이스 악보 초안(Draft) 작성 보조 툴**입니다. (완전한 출판용 정본 악보 제작 도구가 아닙니다.)
> - **개인 사용 한정 (사적이용 복제):** 본 도구로 추출한 악보 및 **Browser Use / Computer Use(유튜브 영상 화면 캡처 보정)** 결과물은 **대한민국 저작권법 제30조(사적이용을 위한 복제)** 범위 내 **개인 연습·연구 목적**으로만 사용해야 합니다.
> - **사용자 책임 명시:** 저작권이 있는 음원을 바탕으로 악보를 생성하거나, 타인의 유튜브 TAB/편곡(저작권법 제5조 **2차적저작물**)을 참조·보정하여 생성한 악보를 **블로그·카페·커뮤니티·오픈채팅 등에 업로드·공유·배포·전송·판매함으로써 발생하는 모든 저작권 분쟁 및 법적 책임은 전적으로 사용자 본인에게 있습니다.**
> - **오탐 방지 방침:** 화음/더블스톱 추정은 오탐 위험이 있어 의도적으로 제외하고, 단선율 기반 베이스 라인 정확도와 반복 마디 안정화에 집중합니다.

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

## 초보자용 빠른 설치 가이드 (Step-by-Step)

### 0단계. 사전 준비물 설치
터미널(Windows PowerShell 또는 macOS 터미널)에서 아래 필수 프로그램을 먼저 설치합니다.

- **Python 3.10 ~ 3.12** (Python 3.12 권장)
  - Windows: `winget install Python.Python.3.12`
  - macOS: `brew install python@3.12`
- **FFmpeg** (유튜브 음원·MP3·M4A 변환 필수)
  - Windows: `winget install Gyan.FFmpeg` (설치 후 터미널을 재시작하세요)
  - macOS: `brew install ffmpeg`
- **(선택) MuseScore 4** (`.musicxml`을 MuseScore 전용 `.mscz` 파일로 자동 변환할 때 필요)
  - Windows: `winget install MuseScore.MuseScore`
  - macOS: `brew install --cask musescore`

### 1단계. 저장소 내려받기 및 가상환경 설치

> **주의:** `pip install`은 반드시 이 프로젝트 폴더(`bass-sound-to-muse-score`) 안으로 `cd`한 뒤에 실행하세요.

```powershell
git clone https://github.com/SuranS2/bass-sound-to-muse-score.git
cd bass-sound-to-muse-score

# 가상환경(.venv) 생성 및 고품질 ML 엔진(Demucs + Librosa pYIN) 설치
python -m venv .venv
.venv\Scripts\pip install -e ".[ml]"

# (선택) 원음 기반 다운비트 정밀 추적 Beat This! 추가 설치 (CPJKU/beat_this, MIT)
# .venv\Scripts\pip install -e ".[ml,beat]"
```

*(macOS / Linux에서는 마지막 줄 대신 `.venv/bin/pip install -e ".[ml]"`를 사용하세요. 가벼운 기본 모드만 필요하면 `-e .`로 설치할 수 있으며, Python 3.10~3.11에서는 `basic-pitch`도 함께 설치되고 Python 3.12+에서는 자동으로 `demucs` + `librosa:pyin` 엔진으로 동작합니다.)*

### 2단계. (선택) MuseScore 4 실시간 연동 플러그인
1. [`plugins/musescore-mcp-websocket.qml`](plugins/musescore-mcp-websocket.qml) 파일을 `%USERPROFILE%\Documents\MuseScore4\Plugins\` 폴더에 복사합니다.
2. MuseScore 4 → **Plugins → Plugin Manager**에서 활성화한 뒤 실행합니다 (`ws://127.0.0.1:8765`, 로컬 루프백 전용).

---

## AI 툴별 MCP 서버 등록 방법 (복사·붙여넣기용)

아래 예시의 `C:\\dev\\bass-sound-to-muse-score` 경로만 **본인이 `git clone`한 실제 폴더 경로**로 바꿔 넣은 뒤, 해당 AI 툴을 **재시작**하세요.

### 1. OpenAI Codex (CLI / Desktop / VS Code 확장)
Codex는 JSON이 아닌 **TOML 문법**을 사용합니다.
- **설정 파일 위치:**
  - Windows: `C:\Users\<사용자이름>\.codex\config.toml`
  - macOS / Linux: `~/.codex/config.toml`

```toml
[mcp_servers.bass-sound-to-musescore]
command = "C:\\dev\\bass-sound-to-muse-score\\.venv\\Scripts\\python.exe"
args = ["-m", "bass_mcp"]

[mcp_servers.bass-sound-to-musescore.env]
BASS_MCP_OUTPUT_DIR = "C:\\dev\\bass-sound-to-muse-score\\output"
```

### 2. Claude Desktop (클로드 데스크톱 앱)
- **설정 파일 위치:**
  - Windows: `%APPDATA%\Claude\claude_desktop_config.json` (주소창에 붙여넣기 가능)
  - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

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

### 3. Claude Code (CLI)
터미널에서 아래 한 줄을 실행하면 바로 등록됩니다.
```powershell
claude mcp add --transport stdio bass-sound-to-musescore -e BASS_MCP_OUTPUT_DIR=C:\dev\bass-sound-to-muse-score\output -- C:\dev\bass-sound-to-muse-score\.venv\Scripts\python.exe -m bass_mcp
```

### 4. Google Antigravity (AGY) & Cursor
- **Google Antigravity 설정 파일 위치:**
  - 전역: `C:\Users\<사용자이름>\.gemini\antigravity\mcp_config.json` (또는 `~/.gemini/config/mcp_config.json`)
  - 프로젝트별: `.agents/mcp_config.json`
- **Cursor 설정 파일 위치:** `C:\Users\<사용자이름>\.cursor\mcp.json` (또는 `~/.cursor/mcp.json`)

[`mcp_config.example.json`](mcp_config.example.json)과 동일한 JSON 형식입니다:
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

*(macOS/Linux에서는 `"command"` 경로를 `/path/to/bass-sound-to-muse-score/.venv/bin/python`으로 바꾸세요. 선택 보안 옵션으로 `BASS_MCP_ALLOWED_DIRS` 환경변수에 `;`로 구분된 허용 폴더를 지정하면 그 외 로컬 음원 접근을 차단합니다.)*

---

## 처음 사용하는 분을 위한 대화 예시 (Quick Start)

1. **연결 확인:** AI 채팅창에 아래와 같이 입력해 `demucs`, `librosa`, `ffmpeg`, `MuseScore4` 인식을 확인합니다.
   > `베이스 MCP 환경 상태 확인해줘`
2. **유튜브 또는 음원 파일로 타브 악보 요청:**
   > `https://www.youtube.com/watch?v=Pxn4J9DyQZA 이 곡 베이스 타브 악보 뽑아줘`
3. **곡명·원본 BPM·박자표·도돌이표 압축·화면 보정 확인 응답:**
   AI가 조회한 곡명과 원본 BPM(예: `SPYAIR - Some Like It Hot, 113 BPM`), 추천 박자표(예: `4/4`)를 먼저 확인해 주며 다음 항목을 함께 질문합니다:
   - **`4/4`·`3/4`·`6/8` 외에 `5/4`·`7/8`·`12/8` 같은 변칙/복합박자는 `time_signature` 직접 지정 필요**
   - **도돌이표 압축 여부 (`compress_repeats`):** 기본값은 풀 스코어(`False`, 전체 마디 전개)이며, 반복 마디를 도돌이표(`|xN`)로 줄이려면 `True` 선택
   - **(선택, 개인 사용 한정) Browser Use / Computer Use 화면 보정:** AI 툴에 브라우저/화면 제어 기능이 있다면 유튜브 영상 TAB을 참고(`prepare_youtube_tab_frames` → `visual_tab_hints`)하여 프렛 위치·전주 쉼표·슬라이드를 보정 가능
   > `응 113 BPM, 4/4 풀스코어로 그대로 변환해줘` *(예: `118 BPM, 7/8 박자, 도돌이표 압축 켜고 뽑아줘`)*
4. **결과물 확인:** `output/<곡명>-<가수>-.../` 폴더에 생성된 `bass_score.mscz` (또는 `bass_score.musicxml`, `bass_line.mid`)를 **MuseScore 4**로 열면 전주/간주 쉼표·스타카토·슬라이드가 반영된 오선보 + 4현/5현 TAB + 코드 기호를 바로 확인할 수 있습니다.

---

## 제공 MCP Tools

| Tool | 설명 |
|---|---|
| `transcribe_bass_pipeline` | 유튜브 URL 또는 로컬 음원 → 곡명/가수·BPM·박자표·도돌이표 압축 확인 요청(`needs_bpm_confirmation`) 후 코드/TAB/MusicXML/MIDI/MSCZ 생성 |
| `lookup_song_bpm` | 허용된 공개 음악 API(`iTunes`/`Deezer`)로 곡명-가수 정규화 및 원본 BPM 조회 |
| `separate_bass_track` | 유튜브 URL 또는 음원에서 베이스 트랙 WAV만 분리 |
| `audio_to_bass_score` | 이미 분리된 베이스 오디오를 확정된 BPM·박자표·`compress_repeats`·`visual_tab_hints`로 악보 변환 |
| `prepare_youtube_tab_frames` | *(개인 연습 보정용)* 유튜브 영상에서 제한된 수(`1~24`장)의 TAB 키프레임을 추출하고 `visual_tab_hints` 스키마·Browser/Computer Use 가이드를 반환 |
| `musescore_live_push` | 생성된 노트 JSON 또는 MusicXML을 실행 중인 MuseScore 4로 전송 (`127.0.0.1:8765`) |
| `check_environment_status` | `ffmpeg`, `yt-dlp`, `demucs`, `basic-pitch`, `beat_this`, MuseScore CLI 설치 상태 및 라이선스 정보 확인 |

---

## Public 공개 가능 여부 및 오픈소스 라이선스 안내

- **GitHub Public 저장소 공개 가능 여부:** **공개 가능(안전)**. 본 프로젝트 코드는 **MIT License**이며, 핵심 Python 의존성도 모두 MIT/BSD/ISC/Apache-2.0/Unlicense 등 퍼미시브 라이선스입니다. GPL-3.0인 **MuseScore 4**와 **FFmpeg**는 코드를 링크하거나 동봉하지 않고 사용자 PC에 설치된 별도 실행 파일을 외부 CLI 서브프로세스로만 호출하므로(FSF GPLv3 FAQ의 별개 프로그램 기준), 본 저장소 코드에 GPL이 전파되지 않습니다.
- **Demucs (`facebookresearch/demucs`) 라이선스 정밀 검토 결과:**
  - **소스 코드 및 추론 프레임워크:** **MIT License** (상업·개인·수정·배포 허용).
  - **사전학습 모델 가중치(`htdemucs`, `mdx` 등):** MIT 라이선스 적용 대상이 아니며, **MUSDB18 / MUSDB18-HQ** 등 연구·비상업 데이터셋 약관의 영향을 받아 업스트림 메인테이너 기준 **개인·비상업·연구 목적(non-commercial / research only)**으로 취급됩니다. 본 저장소에는 Demucs 가중치를 직접 동봉하지 않으며, 사용자가 선택 설치(`[ml]`) 시 로컬 환경에서만 내려받아 개인 용도로 실행합니다. 상업용 제품/서비스에 `htdemucs` 가중치를 번들링·배포하려면 별도 법률·라이선스 검토 또는 자체 학습 모델 교체가 필요합니다.
- **공개 시 주의사항:**
  - `.gitignore`에 이미 포함된 `output/`, `*.wav`, `*.mp3`, `*.mid`, `*.musicxml`, `*.mscz`, `*.jpg` 등 **유튜브 다운로드 음원·캡처 프레임 및 생성 악보 파일(원곡 저작물 파생물)과 Demucs 사전학습 모델 가중치는 저장소에 커밋하지 마세요.**
  - 자세한 전문과 출처 고지는 [LICENSE](LICENSE), [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)를 참고하세요.

- **Spotify Basic Pitch** (`spotify/basic-pitch`) — Apache License 2.0
- **Meta / Facebook Research Demucs** (`facebookresearch/demucs`) — 코드 MIT License / 사전학습 가중치(`htdemucs` 등) 비상업·연구·개인 목적 한정
- **Beat This!** (`CPJKU/beat_this`) — 코드 및 모델 가중치 MIT License (학습 원음 데이터셋 별도 저작권 유의)
- **librosa** (`librosa/librosa`) — ISC License
- **NumPy / SciPy / websockets** — BSD 3-Clause License
- **ghchen99/mcp-musescore** (`ghchen99/mcp-musescore`) — MIT License
- **sandraschi/stems-mcp** (`sandraschi/stems-mcp`) — MIT License
- **pretty_midi** (`craffel/pretty-midi`) — MIT License
- **yt-dlp** (`yt-dlp/yt-dlp`) — The Unlicense
- **MCP Python SDK** (`modelcontextprotocol/python-sdk`) — MIT License
