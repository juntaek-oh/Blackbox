# 🚗 AI 스마트 블랙박스 시스템

하이브리드 블랙박스 영상  
![GIF 1](https://github.com/user-attachments/assets/55caa8fc-b178-4640-9e15-2ad3bab294ad)  
![GIF 2](https://github.com/user-attachments/assets/f832f5c1-f685-4ade-97a5-10bc61b4dd2e)

웹캠 블랙박스 영상  
![GIF 3](https://github.com/user-attachments/assets/94f06c32-7f0b-455c-80cd-50aec259a48d)  
![GIF 4](https://github.com/user-attachments/assets/dd0ed0d7-2f6c-44e8-a775-a90fcd987fd3)

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![YOLOv4](https://img.shields.io/badge/YOLOv4-Tiny-red.svg)](https://github.com/AlexeyAB/darknet)

**실시간 IoU 기반 차량 추적 및 앞차 출발 감지 시스템**  
*"앞차가 출발했습니다" - 신호대기 중 앞차 출발을 AI가 자동으로 감지하여 TTS 음성으로 안내하는 스마트 블랙박스"*

[🎯 주요 기능](#-주요-기능) •
[🚀 빠른 시작](#-설치-및-실행) •
[📖 사용법](#-사용법) •
[⚙️ 설정](#️-설정) •
[🔧 문제해결](#-문제해결)
</div>

---

## 📋 목차

- [🎯 주요 기능](#-주요-기능)
- [🏗️ 시스템 아키텍처](#-시스템-아키텍처)
- [🚀 설치 및 실행](#-설치-및-실행)
- [📖 사용법](#-사용법)
- [⚙️ 설정](#️-설정)
- [🎛️ 키 조작법](#️-키-조작법)
- [📊 성능 정보](#-성능-정보)
- [🔧 문제해결](#-문제해결)
- [🛠️ 개발 정보](#️-개발-정보)
- [🤝 기여하기](#-기여하기)
- [📄 라이선스](#-라이선스)
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
- 라즈베리파이 전용 설정 파일 지원

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

실시간 모드: 웹캠 → AI 분석 → TTS 알림
분석 모드: YouTube/MP4 → 시간 조절 → AI 분석

text

---

### 📁 프로젝트 구조

Blackbox/
├── black_box_webcam.py # PC/웹캠 실시간 모드
├── black_box_raspberrypi.py # 라즈베리파이 v4l2 최적화 모드
├── hybrid_blackbox.py # YouTube/MP4 분석 모드
├── tts_config.py # TTS 음성 설정
├── tts_settings.py # TTS 세부 설정
├── webcam_blackbox_config.json # PC/웹캠 설정
├── raspberrypi_blackbox_config.json # 라즈베리파이 설정
├── hybrid_blackbox_config.json # 하이브리드 설정
├── requirements.txt # 의존성 패키지
├── README.md # 설명 문서
├── downloads/ # 다운로드된 영상 (자동 생성)
├── webcam_output/ # 녹화 파일 (자동 생성)
├── log/ # 로그 파일 (자동 생성)
├── model/ # YOLO 가중치 및 설정 파일

text

---

## 🚀 설치 및 실행

### 📋 시스템 요구사항

| 구분    | PC/웹캠          | 라즈베리파이          |
|---------|------------------|----------------------|
| CPU     | Intel i3 이상    | Raspberry Pi 4 이상  |
| RAM     | 4GB 이상         | 2GB 이상             |
| GPU     | 권장 아님        | -                   |
| Python  | 3.8 이상         | 3.8 이상             |

---

### 설치 과정

1. 저장소 복제  
git clone https://github.com/juntaek-oh/Blackbox.git
cd Blackbox

text
2. 의존성 설치  
pip install -r requirements.txt

text
3. AI 모델 다운로드 및 위치 지정  
wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names

text
다운로드한 파일은 `./model/` 폴더에 저장
4. 설정 파일 환경별 수정  
 - PC/웹캠: `webcam_blackbox_config.json`
 - 라즈베리파이: `raspberrypi_blackbox_config.json`
 - 하이브리드: `hybrid_blackbox_config.json`

---

### ▶️ 실행 방법

- PC/웹캠 실시간 모드  
python black_box_webcam.py --config webcam_blackbox_config.json

text
- Raspberry Pi v4l2 실시간 모드  
python black_box_raspberrypi.py --config raspberrypi_blackbox_config.json

text
- YouTube/MP4 분석 모드  
python hybrid_blackbox.py --config hybrid_blackbox_config.json

text

---

## 📖 사용법

### 웹캠 실시간 모드

실시간으로 웹캠을 통한 앞차 출발 감지 및 녹화

┌─────────────────────────────────────────────┐
│ 🔴 REC 14:30:25 | FPS: 28.5 | 추적: 3 | 출발: 1 │
├─────────────────────────────────────────────┤
│ 🚗 [ID-1 WAITING] ────┐ 🚦 RED │
│ 🚗 [ID-2 MOVING] ─────┤ │
│ 🚙 [ID-3 DEPARTED] ───┘ │
│ ═══════ 차선 중심선 ═══════ │
└─────────────────────────────────────────────┘

text
- [SPACE]: 일시정지/재생 토글  
- [S]: 현재 화면 스크린샷 저장  
- [Q]: 프로그램 종료  
- [ESC]: 긴급 종료  
- [R]: 웹캠 재연결

### YouTube/MP4 분석 모드

화살표 키 등으로 시간 조절하며 영상 분석하여 출발 패턴 학습

---

## ⚙️ 설정

### PC/웹캠 설정 (`webcam_blackbox_config.json`)

{
"camera": {
"device_id": 0,
"width": 1280,
"height": 720,
"fps": 30
},
"detection": {
"detection_interval": 1,
"confidence_threshold": 0.5
}
}

text

### 라즈베리파이 설정 (`raspberrypi_blackbox_config.json`)

{
"camera": {
"device_id": "auto",
"width": 1280,
"height": 720,
"fps": 30,
"buffer_size": 1
},
"detection": {
"detection_interval": 2,
"confidence_threshold": 0.5
}
}

text

### 하이브리드 설정 (`hybrid_blackbox_config.json`)

{
"youtube": {
"url": "your_default_url",
"quality": "720p"
},
"tracking": {
"movement_threshold": 1.5,
"iou_threshold": 0.25
}
}

text

---

## 🎯 핵심 알고리즘 파라미터

출발감지_대기시간 = 1초 # 2초 → 1초 단축
정지상태_판정기준 = 2.5픽셀 # 1.5px → 2.5px 완화
정지비율_임계값 = 50% # 70% → 50% 완화
움직임감지_임계값 = 2.25픽셀 # 4px → 2.25px 민감화

text

---

## 🚦 신호등 HSV 색상 범위 (한국 신호등 최적화)

빨간불: [0-10, 170-180] + [100-255, 100-255]
노란불: [18-35] + [120-255, 120-255]
초록불: [45-90] + [100-255, 100-255]

text

---

## 🛣️ 차선 감지 알고리즘

- Canny Edge Detection → 차선 엣지 추출  
- Hough Transform → 직선 패턴 감지  
- 동적 ROI → 화면 하단 65% 영역 집중 분석  
- 중심점 계산 → 좌우 차선 중심의 Front Zone 설정

---

## 📊 성능 정보

| 해상도      | CPU 사용률 | 평균 FPS | 감지 지연      |
|-------------|------------|----------|---------------|
| 640x480     | 약 30%     | 28~30    | 0.1초 미만    |
| 1280x720    | 약 45%     | 25~28    | 0.2초 미만    |
| 1920x1080   | 약 65%     | 20~25    | 0.3초 미만    |

---

## 🔧 문제해결

- **모델 파일 없음 오류**
    - `FileNotFoundError: yolov4-tiny.weights`
    - 반드시 yolov4-tiny.weights, cfg, coco.names 다운 및 올바른 위치 지정
- **웹캠 연결 실패**
    - `cv2.error: Cannot open camera`
    - `webcam_blackbox_config.json`에서 `device_id` 변경하거나 장치 상태 확인
- **성능 저하 문제**
    - 해상도 및 FPS 낮추거나 `detection_interval` 증가로 AI 분석 주기 조절
- **YouTube 다운로드 오류**
    - `pip install --upgrade yt-dlp` 실행

---

## 🛠️ 개발 정보

#### 최근 업데이트 (v2.0, 2025년)
- 강화된 USB 모니터링 및 긴급종료 키(`ESC`, `Q`) 추가
- 하이브리드 시스템: YouTube + MP4 통합
- 출발감지 최적화: 1초 대기, 50% 정지율
- Downloads 폴더 통합 관리

#### 향후 계획
- GPU 가속 (CUDA/OpenCL)
- 클라우드 연동 (AWS, GCP)
- 모바일 앱 원격 제어
- 다중 카메라 동시 처리
- 고급 AI 기능 (차선 변경 감지, 졸음운전 감지 등)

---

## 🤝 기여하기

1. 프로젝트 Fork  
2. Feature Branch 생성 (`git checkout -b feature/AmazingFeature`)  
3. 커밋 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치 푸시 (`git push origin feature/AmazingFeature`)
5. Pull Request 생성

---

## 🐛 버그 리포트

Issues 탭에 OS, Python 버전, 에러 메시지 전체, 재현 단계 포함해 제출해주세요.

---

## 📄 라이선스

MIT License (LICENSE 파일 참조)

---

## 📞 연락처

- 이메일: [ojt8416@gmail.com](mailto:ojt8416@gmail.com)
- GitHub Issues: [링크](https://github.com/juntaek-oh/Blackbox/issues)
- Wiki: [링크](https://github.com/juntaek-oh/Blackbox/wiki)

---

<div align="center">

## 🚗 안전한 운전의 시작, AI 블랙박스 시스템과 함께하세요! 🚗  
**실시간 차량 추적과 앞차 출발 감지를 합리적 성능과 사용성으로 구현한 스마트 AI 블랙박스**  
⭐ 도움 되셨다면 Star 부탁드립니다! ⭐  

</div>
