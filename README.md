# 🚗 AI 블랙박스 시스템 (Vehicle Blackbox AI System)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()

**실시간 IoU 기반 차량 추적 및 앞차 출발 감지 시스템**

> 🎯 **"앞차가 출발했습니다"** - 신호대기 중 앞차 출발을 AI가 자동으로 감지하여 TTS 음성으로 알려주는 스마트 블랙박스

---

## 🌟 핵심 기능

### 🤖 **AI 기반 실시간 추적**
- **IoU 알고리즘**: 정밀한 차량 추적 및 ID 관리
- **차선 기반 필터링**: 동적 ROI로 앞차만 정확히 감지
- **출발 감지 최적화**: 1초 대기, 50% 정지율로 빠른 반응

### 🎛️ **다중 입력 지원**
- **웹캠 모드**: USB 캠 실시간 녹화 + AI 분석
- **YouTube 스트림**: 실시간 스트리밍 (빠른 시작)
- **YouTube 다운로드**: 완전한 시간 조절 지원
- **MP4 재생**: 로컬 파일 분석

### 🚦 **고급 감지 시스템**
- **신호등 색상 인식**: 한국 신호등 최적화 HSV 필터링
- **차선 감지**: Hough Transform + 동적 중심점 계산
- **움직임 분석**: 정지 → 출발 패턴 학습

### 📢 **스마트 알림**
- **TTS 음성 안내**: "앞차가 출발했습니다" 실시간 알림
- **시각적 피드백**: 실시간 추적 상태 표시
- **로그 시스템**: 상세한 이벤트 기록

---

## 🎮 시스템 구성

### 📹 **웹캠 블랙박스** (`black_box_webcam.py`)
🔴 실시간 녹화 → 🤖 AI 분석 → 📢 TTS 알림
- USB 웹캠 실시간 처리
- 자동 세그먼트 녹화 (10분 단위)
- 실시간 앞차 출발 감지

### 🎬 **하이브리드 블랙박스** (`hybrid_blackbox.py`)
📺 YouTube URL → 📥 다운로드/스트림 → 🎛️ 시간 조절 → 🤖 AI 분석
- **3가지 모드**: 스트림/다운로드/파일
- **화살표 키 시간 조절**: ⬅️➡️ 점프, ⬆️⬇️ 10초, A/D 1초
- **Downloads 폴더 통합 관리**

---

## 🚀 빠른 시작

### 1️⃣ **설치**
git clone https://github.com/your-username/vehicle-blackbox-system.git
cd vehicle-blackbox-system
pip install -r requirements.txt

### 2️⃣ **AI 모델 다운로드**
자동 다운로드 스크립트 (Windows/Linux/macOS)
cd models
curl -L https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights -o yolov4-tiny.weights
curl -L https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg -o yolov4-tiny.cfg
curl -L https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names -o coco.names

### 3️⃣ **실행**
웹캠 실시간 블랙박스
python black_box_webcam.py

하이브리드 블랙박스 (YouTube + MP4)
python hybrid_blackbox.py

---

## 🎯 사용법 가이드

### 🎮 **공통 조작키**
| 키 | 기능 | 설명 |
|---|---|---|
| `Q` | 종료 | 프로그램 안전 종료 |
| `SPACE` | 일시정지 | 재생/일시정지 토글 |
| `S` | 스크린샷 | 현재 화면 저장 |
| `ESC` | 긴급종료 | 강제 종료 (USB 연결 해제시) |

### 🎛️ **하이브리드 모드 전용 (시간 조절)**
| 키 | 기능 | 설명 |
|---|---|---|
| `←` `→` | 점프 | 설정된 시간만큼 이동 |
| `↑` `↓` | 10초 점프 | 빠른 이동 |
| `A` `D` | 1초 미세조정 | 정밀 탐색 |
| `T` | 점프시간 변경 | 5s → 10s → 30s → 60s... |

### 📂 **파일 관리**
project/
├── downloads/ # YouTube 다운로드 + 수동 MP4 파일
├── webcam_output/ # 웹캠 녹화 파일 (10분 세그먼트)
├── log/ # 시스템 로그 + TTS 텍스트
└── config/ # 설정 파일

---

## ⚙️ 설정 가이드

### 📹 **웹캠 최적화 (`config/webcam_config.json`)**
{
"camera": {
"device_id": 0, // USB 웹캠 번호 (0, 1, 2...)
"width": 1280, // 해상도 (높을수록 정밀, 낮을수록 빠름)
"height": 720,
"fps": 30 // 프레임율 (높을수록 부드럽, 낮을수록 안정)
},
"detection": {
"detection_interval": 1, // N프레임마다 AI 분석 (1=매프레임, 2=2프레임마다)
"confidence_threshold": 0.5
}
}
### 🎬 **하이브리드 최적화 (`config/hybrid_config.json`)**
{
"youtube": {
"url": "your_default_url",
"quality": "720p" // 480p, 720p, 1080p, best
},
"tracking": {
"movement_threshold": 1.5, // 출발감지 민감도 (낮을수록 민감)
"iou_threshold": 0.25 // 추적 정확도 (높을수록 정확)
}
}
---

## 🎨 UI 미리보기

### 🔴 **웹캠 모드 - 실시간 처리**
┌─────────────────────────────────────────────────┐
│ 🔴 REC 14:30:25 | FPS: 28.5 | 추적: 3 | 출발: 1 │
├─────────────────────────────────────────────────┤
│ │
│ 🚗 [ID-1 WAITING] ────┐ │
│ │ 🚦 RED │
│ 🚗 [ID-2 MOVING] ─────┤ │
│ │ │
│ 🚙 [ID-3 DEPARTED] ───┘ │
│ │
│ ═══════ 차선 중심선 ═══════ │
└─────────────────────────────────────────────────┘
[SPACE] 일시정지 [S] 스크린샷 [Q] 종료

### 🎬 **하이브리드 모드 - YouTube/MP4**
┌─────────────────────────────────────────────────┐
│ 📺 YouTube Mode | 15:30 / 45:20 (34%) | SEEK: ON │
├─────────────────────────────────────────────────┤
│ │
│ 📥 다운로드 완료! │
│ 🎛️ 화살표 키 시간조절 가능 │
│ │
│ 🚗 WAIT: 2 🚗 MOVE: 1 🚗 DEPT: 3 │
│ 🚦 GREEN │
└─────────────────────────────────────────────────┘
[ARROWS] 시간이동 [T] 점프설정 [SPACE] 일시정지 [Q] 종료

---

## 🧠 AI 알고리즘 상세

### 🎯 **IoU 기반 차량 추적**
핵심 알고리즘 파라미터 (최적화 완료)
출발감지_대기시간 = 1초 # 기존 2초 → 1초로 단축
정지상태_판정기준 = 2.5픽셀 # 기존 1.5px → 2.5px로 완화
정지비율_임계값 = 50% # 기존 70% → 50%로 완화
움직임감지_임계값 = 2.25픽셀 # 기존 4px → 2.25px로 민감화

### 🛣️ **동적 차선 감지**
1. **Canny Edge Detection** → 차선 엣지 추출
2. **Hough Transform** → 직선 패턴 감지
3. **동적 ROI** → 화면 하단 65% 영역 집중 분석
4. **중심점 계산** → 좌우 차선 중심의 Front Zone 설정

### 🚦 **신호등 색상 인식**
HSV 색상 범위 (한국 신호등 최적화)
빨간불: [0-10, 170-180] + [100-255, 100-255]
노란불: [18-35] + [120-255, 120-255]
초록불: [45-90] + [100-255, 100-255]
---

## 📊 성능 벤치마크

### 💻 **권장 시스템 사양**
| 구분 | 최소 | 권장 | 고성능 |
|---|---|---|---|
| **CPU** | Intel i3 | Intel i5 | Intel i7+ |
| **RAM** | 4GB | 8GB | 16GB+ |
| **GPU** | 통합 | GTX 1050 | RTX 2060+ |
| **해상도** | 720p | 1080p | 4K |

### ⚡ **실시간 성능**
| 해상도 | CPU 사용률 | 평균 FPS | 감지 지연 |
|---|---|---|---|
| **640x480** | ~30% | 28-30 | <0.1초 |
| **1280x720** | ~45% | 25-28 | <0.2초 |
| **1920x1080** | ~65% | 20-25 | <0.3초 |

---

## 🔧 문제해결 가이드

### ❌ **일반적인 문제들**

#### 🔴 **"AI 모델 파일이 없습니다"**
해결방법: 모델 파일 다운로드
cd models
wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
#### 🔴 **"웹캠을 찾을 수 없습니다"**
// config/webcam_config.json에서 device_id 변경
{
"camera": {
"device_id": 1 // 0 → 1로 변경 후 재시도
}
}
#### 🔴 **"FPS가 너무 낮습니다"**
// 성능 최적화 설정
{
"camera": {
"width": 640, // 해상도 낮춤
"height": 480,
"fps": 15 // 목표 FPS 낮춤
},
"detection": {
"detection_interval": 3 // AI 분석 주기 늘림
}
}
#### 🔴 **"USB 웹캠 뽑으면 프로그램이 멈춤"**
- **해결됨!** ✅ 강화된 USB 모니터링 및 긴급종료 키 (`ESC`, `Q`) 추가

#### 🔴 **"YouTube 다운로드 실패"**
pip install --upgrade yt-dlp
### 🆘 **긴급 상황**
| 상황 | 해결 키 | 설명 |
|---|---|---|
| 프로그램 먹통 | `ESC` | 강제 종료 |
| USB 캠 연결 해제 | `Q` | 안전 종료 |
| 카메라 재연결 | `R` | 웹캠 재연결 시도 |

---

## 📈 업데이트 로드맵

### 🎯 **v5.0 (계획)**
- [ ] **GPU 가속**: CUDA/OpenCL 지원으로 실시간 4K 처리
- [ ] **클라우드 연동**: AWS/GCP 실시간 스트리밍
- [ ] **모바일 앱**: 스마트폰 원격 제어
- [ ] **다중 카메라**: 전후방 동시 처리

### 🔥 **v4.3 (현재)**
- [x] **하이브리드 시스템**: YouTube + MP4 통합
- [x] **USB 안전성**: 연결 해제 시 안전 종료
- [x] **출발감지 최적화**: 1초 대기, 50% 정지율
- [x] **Downloads 폴더 통합**: 파일 관리 간소화

---

## 🤝 기여하기

### 📋 **기여 방법**
1. **Fork** the Project
2. **Create** Feature Branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** Changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to Branch (`git push origin feature/AmazingFeature`)
5. **Open** Pull Request

### 🐛 **버그 리포트**
Issues 탭에서 다음 정보와 함께 버그를 리포트해주세요:
- **OS**: Windows 10/11, Ubuntu 20.04, macOS 등
- **Python 버전**: 3.8, 3.9, 3.10 등
- **에러 메시지**: 전체 traceback
- **재현 단계**: 1-2-3 단계

### 💡 **기능 제안**
- **실시간 차선 변경 감지**
- **졸음운전 감지**
- **스마트폰 미러링**
- **블랙박스 영상 자동 편집**

---

## 📄 라이센스

이 프로젝트는 **MIT License** 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.

---

## 👨‍💻 개발자

**AI 블랙박스 시스템** - 실시간 차량 추적의 새로운 패러다임

- 🎯 **핵심 기술**: IoU 추적, 동적 차선 감지, HSV 신호등 인식
- 🚀 **성능 최적화**: 1초 출발감지, 30fps 실시간 처리
- 🎛️ **사용자 경험**: 화살표 키 시간조절, TTS 음성안내

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/vehicle-blackbox-system&type=Date)](https://star-history.com/#your-username/vehicle-blackbox-system&Date)

---

**⭐ 이 프로젝트가 도움이 되셨다면 Star를 눌러주세요!**

**🔄 Pull Requests와 Issues를 환영합니다!**

**📧 문의: your-email@example.com**

---

<div align="center">

**🚗 안전한 운전의 시작, AI 블랙박스 시스템과 함께하세요! 🚗**

Made with ❤️ by **AI 블랙박스 팀**

</div>
