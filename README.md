# 🚗 AI 스마트 블랙박스 시스템

<table>
<tr>
<td width="50%">

### 하이브리드 블랙박스 영상
![GIF 1](https://github.com/user-attachments/assets/55caa8fc-b178-4640-9e15-2ad3bab294ad)
![GIF 2](https://github.com/user-attachments/assets/f832f5c1-f685-4ade-97a5-10bc61b4dd2e)

</td>
<td width="50%">

### 웹캠 블랙박스 영상
![GIF 3](https://github.com/user-attachments/assets/94f06c32-7f0b-455c-80cd-50aec259a48d)
![GIF 4](https://github.com/user-attachments/assets/dd0ed0d7-2f6c-44e8-a775-a90fcd987fd3)

</td>
</tr>
</table>

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![YOLOv4](https://img.shields.io/badge/YOLOv4-Tiny-red.svg)](https://github.com/AlexeyAB/darknet)

**실시간 IoU 기반 차량 추적 및 앞차 출발 감지 시스템**

*"앞차가 출발했습니다" - 신호대기 중 앞차 출발을 AI가 자동으로 감지하여 TTS 음성으로 안내하는 스마트 블랙박스*

</div>

---

## 📋 목차

- [🎯 주요 기능](#-주요-기능)
- [🏗️ 시스템 아키텍처](#️-시스템-아키텍처)
- [🚀 설치 및 실행](#-설치-및-실행)
- [📖 사용법](#-사용법)
- [⚙️ 설정](#️-설정)
- [🎛️ 키 조작법](#️-키-조작법)
- [📊 성능 정보](#-성능-정보)
- [🔧 문제해결](#-문제해결)
- [🛠️ 개발 정보](#️-개발-정보)
- [🤝 기여하기](#-기여하기)
- [📞 연락처](#-연락처)

---

## 🎯 주요 기능

### 🔴 실시간 처리 모드
- **웹캠 녹화**: USB 웹캠을 통한 실시간 녹화 및 AI 분석
- **자동 세그먼트**: 10분 단위 자동 분할 저장
- **실시간 TTS**: 앞차 출발 시 즉시 음성 안내

### 🚗 라즈베리파이 지원
- **v4l2 기반 카메라 드라이버** 자동 탐지 및 최적화
- 저전력 환경 맞춤 해상도 및 프레임 설정 (기본 1280x720, 30fps)
- 라즈베리파이 전용 설정파일 지원

### 📺 YouTube/MP4 분석 모드
- **3가지 입력 방식**: YouTube 스트리밍, 다운로드, 로컬 MP4 파일
- **시간 제어**: 화살표 키를 통한 정밀 시간 조절
- **완전 분석**: 기록된 영상의 상세 분석

### 🤖 AI 핵심 기술
- **IoU 알고리즘**: 정밀한 차량 추적 및 ID 관리
- **차선 기반 필터링**: 동적 ROI로 앞차만 정확히 감지
- **출발 감지 최적화**: 1초 대기, 50% 정지율로 빠른 반응
- **신호등 색상 인식**: 한국 신호등 최적화 HSV 필터링
- **차선 감지**: Canny Edge + Hough Transform, 동적 Front Zone 계산

### 🎛️ 사용자 인터페이스
- **시각적 피드백**: 실시간 추적 상태 표시
- **로그 시스템**: 상세한 이벤트 기록
- **직관적 조작**: 키보드 단축키로 모든 기능 제어

---

## 🏗️ 시스템 아키텍처

```
🔴 실시간 모드: 웹캠 → AI 분석 → TTS 알림
📺 분석 모드: YouTube/MP4 → 시간 조절 → AI 분석
```

### 📁 프로젝트 구조

```
Blackbox/
├── black_box_webcam.py           # PC/웹캠 실시간 모드
├── black_box_raspberrypi.py      # 라즈베리파이 v4l2 최적화 모드
├── hybrid_blackbox.py            # YouTube/MP4 분석 모드
├── tts_config.py                 # TTS 음성 설정
├── tts_settings.py               # TTS 세부 설정
├── webcam_blackbox_config.json   # PC/웹캠 설정
├── raspberrypi_blackbox_config.json # 라즈베리파이 설정
├── hybrid_blackbox_config.json   # 하이브리드 설정
├── requirements.txt              # 의존성 패키지
├── README.md                     # 설명 문서
├── downloads/                    # 다운로드된 영상 (자동 생성)
├── webcam_output/                # 녹화 파일 (자동 생성)
├── log/                          # 로그 파일 (자동 생성)
└── model/                        # YOLO 가중치 및 설정 파일
```

---

## 🚀 설치 및 실행

### 📋 시스템 요구사항

| 구분    | PC/웹캠          | 라즈베리파이        |
|---------|------------------|-------------------|
| CPU     | Intel i3 이상    | Raspberry Pi 4 이상|
| RAM     | 4GB 이상         | 2GB 이상          |
| GPU     | 권장 아님        | -                 |
| Python  | 3.8 이상         | 3.8 이상          |

### 설치 과정

1. **저장소 복제**
```bash
git clone https://github.com/juntaek-oh/Blackbox.git
cd Blackbox
```

2. **의존성 설치**
```bash
pip install -r requirements.txt
```

3. **AI 모델 다운로드 및 위치 지정**
```bash
wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
```
> 다운로드한 파일은 `./model/` 폴더에 저장

4. **설정 파일 환경별 수정**
- PC/웹캠: `webcam_blackbox_config.json`
- 라즈베리파이: `raspberrypi_blackbox_config.json`
- 하이브리드: `hybrid_blackbox_config.json`

### ▶️ 실행 방법

**PC/웹캠 실시간 모드**
```bash
python black_box_webcam.py --config webcam_blackbox_config.json
```

**Raspberry Pi v4l2 실시간 모드**
```bash
python black_box_raspberrypi.py --config raspberrypi_blackbox_config.json
```

**YouTube/MP4 분석 모드**
```bash
python hybrid_blackbox.py --config hybrid_blackbox_config.json
```

---

## 📖 사용법

### 웹캠 실시간 모드
실시간으로 웹캠을 통한 앞차 출발 감지 및 녹화

```
┌─────────────────────────────────────────────┐
│ 🔴 REC 14:30:25 | FPS: 28.5 | 추적: 3 | 출발: 1 │
├─────────────────────────────────────────────┤
│                                             │
│    🚗 [ID-1 WAITING] ────┐                  │
│    🚦 RED                │                  │
│    🚗 [ID-2 MOVING] ─────┤                  │
│                          │                  │
│    🚙 [ID-3 DEPARTED] ───┘                  │
│                                             │
│    ═══════ 차선 중심선 ═══════               │
└─────────────────────────────────────────────┘
   [SPACE] 일시정지  [S] 스크린샷  [Q] 종료
```

### YouTube/MP4 분석 모드
화살표 키 등으로 시간 조절하며 영상 분석하여
