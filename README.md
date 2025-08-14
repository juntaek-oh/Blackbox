# 🚗 AI 스마트 블랙박스 시스템

<img src="https://github.com/user-attachments/assets/5129212e-a56d-4a65-9e91-40769a18b2fb" width="350"/>

<img src="https://github.com/user-attachments/assets/95b3667e-6938-45e2-a393-8460dc290677" width="350"/>

<img width="1375" height="635" alt="static" src="https://github.com/user-attachments/assets/1942ccea-6a6c-4109-a07f-8d13d3842640" />
<img width="1395" height="784" alt="moving" src="https://github.com/user-attachments/assets/8998dae5-9b65-415e-8f68-ea4b07dda6a7" />


<div align="center">
  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![YOLOv4](https://img.shields.io/badge/YOLOv4-Tiny-red.svg)](https://github.com/AlexeyAB/darknet)

**실시간 IoU 기반 차량 추적 및 앞차 출발 감지 시스템**

*"앞차가 출발했습니다" - 신호대기 중 앞차 출발을 AI가 자동으로 감지하여 TTS 음성으로 알려주는 스마트 블랙박스*

[🎯 주요 기능](#-주요-기능) •
[🚀 빠른 시작](#-설치-및-실행) •
[📖 사용법](#-사용법) •
[⚙️ 설정](#️-설정) •
[🔧 문제해결](#-문제해결)

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
- [📄 라이선스](#-라이선스)
- [📞 연락처](#-연락처)

---

## 🎯 주요 기능

### 🔴 실시간 처리 모드
- **웹캠 녹화**: USB 웹캠을 통한 실시간 녹화 및 AI 분석
- **자동 세그먼트**: 10분 단위 자동 분할 저장
- **실시간 TTS**: 앞차 출발 시 즉시 음성 안내

### 📺 YouTube/MP4 분석 모드  
- **3가지 입력 방식**: YouTube 스트리밍, 다운로드, 로컬 MP4 파일
- **시간 제어**: 화살표 키를 통한 정밀 시간 조절
- **완전 분석**: 기록된 영상의 상세 분석

### 🤖 AI 핵심 기술
- **IoU 알고리즘**: 정밀한 차량 추적 및 ID 관리
- **차선 기반 필터링**: 동적 ROI로 앞차만 정확히 감지  
- **출발 감지 최적화**: 1초 대기, 50% 정지율로 빠른 반응
- **신호등 색상 인식**: 한국 신호등 최적화 HSV 필터링
- **차선 감지**: Hough Transform + 동적 중심점 계산

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
├── 🐍 black_box_webcam.py           # 실시간 웹캠 모드
├── 🐍 hybrid_blackbox.py            # YouTube/MP4 분석 모드
├── 🐍 tts_config.py                 # TTS 음성 설정
├── 🐍 tts_settings.py               # TTS 세부 설정
├── ⚙️ hybrid_blackbox_config.json   # 하이브리드 모드 설정
├── ⚙️ webcam_blackbox_config.json   # 웹캠 설정
├── 📋 requirements.txt              # 의존성 패키지
├── 📖 README.md                     # 프로젝트 문서
├── 📁 downloads/                    # 다운로드된 영상 (자동 생성)
├── 📁 webcam_output/               # 웹캠 녹화 파일 (자동 생성)
└── 📁 log/                         # 로그 파일 (자동 생성)

※ AI 모델 파일은 별도 다운로드 필요:
   - yolov4-tiny.weights (YOLO 가중치)
   - yolov4-tiny.cfg (YOLO 설정)  
   - coco.names (클래스 이름)
```

### 🔄 처리 흐름
1. **영상 입력** → 웹캠/YouTube/MP4
2. **전처리** → 해상도 조정, 프레임 추출
3. **객체 감지** → YOLO를 통한 차량 검출
4. **추적 알고리즘** → IoU 기반 차량 ID 관리
5. **차선 분석** → Hough Transform 차선 검출
6. **ROI 설정** → 앞차 영역 동적 계산
7. **출발 감지** → 움직임 패턴 분석
8. **알림 처리** → TTS 음성 출력

---

## 🚀 설치 및 실행

### 📋 시스템 요구사항

| 구분    | 최소 사양     | 권장 사양     | 고성능       |
|---------|---------------|---------------|--------------|
| **CPU** | Intel i3     | Intel i5     | Intel i7+   |
| **RAM** | 4GB          | 8GB          | 16GB+       |
| **GPU** | 통합 그래픽   | GTX 1050     | RTX 2060+   |
| **Python** | 3.8+      | 3.9+         | 3.10+       |

### 📦 설치 과정

#### 1️⃣ 저장소 복제
```bash
git clone https://github.com/juntaek-oh/Blackbox.git
cd Blackbox
```

#### 2️⃣ 의존성 설치
```bash
pip install -r requirements.txt
```

#### 3️⃣ AI 모델 다운로드
**Windows:**
```bash
curl -L https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights -o yolov4-tiny.weights
curl -L https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg -o yolov4-tiny.cfg  
curl -L https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names -o coco.names
```

**Linux/macOS:**
```bash
wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
```

### ▶️ 실행 방법

#### 🔴 실시간 웹캠 모드
```bash
python black_box_webcam.py
```

#### 📺 YouTube/MP4 분석 모드  
```bash
python hybrid_blackbox.py
```

---

## 📖 사용법

### 🔴 웹캠 실시간 모드

실시간으로 웹캠을 통해 앞차의 출발을 감지합니다.

```
┌─────────────────────────────────────────────────┐
│ 🔴 REC 14:30:25 | FPS: 28.5 | 추적: 3 | 출발: 1 │
├─────────────────────────────────────────────────┤
│                                                 │
│   🚗 [ID-1 WAITING] ────┐                     │
│                          🚦 RED                │
│   🚗 [ID-2 MOVING] ─────┤                     │
│                                                 │  
│   🚙 [ID-3 DEPARTED] ───┘                     │
│                                                 │
│         ═══════ 차선 중심선 ═══════            │
└─────────────────────────────────────────────────┘
[SPACE] 일시정지  [S] 스크린샷  [Q] 종료
```

**주요 기능:**
- 🎬 **자동 녹화**: 10분 단위 세그먼트로 자동 저장
- 🎯 **실시간 추적**: 여러 차량 동시 추적 및 ID 관리  
- 📢 **TTS 알림**: "앞차가 출발했습니다" 즉시 음성 안내
- 📊 **상태 표시**: 실시간 FPS, 추적 차량 수, 출발 횟수 표시

### 📺 YouTube/MP4 분석 모드

YouTube URL이나 MP4 파일을 분석하여 앞차 출발 패턴을 학습합니다.

```
┌─────────────────────────────────────────────────┐
│ 📺 YouTube Mode | 15:30 / 45:20 (34%) | SEEK: ON │
├─────────────────────────────────────────────────┤
│                                                 │
│   📥 다운로드 완료!                             │
│   🎛️ 화살표 키 시간조절 가능                    │
│                                                 │
│   🚗 WAIT: 2  🚗 MOVE: 1  🚗 DEPT: 3          │
│                      🚦 GREEN                   │
└─────────────────────────────────────────────────┘
[ARROWS] 시간이동  [T] 점프설정  [SPACE] 일시정지  [Q] 종료
```

**주요 기능:**
- 🎥 **3가지 모드**: YouTube 스트리밍/다운로드/로컬 MP4 파일
- ⏯️ **시간 제어**: 화살표 키로 정밀 시간 탐색
- 📁 **통합 관리**: Downloads 폴더에서 모든 파일 관리
- 📈 **상세 분석**: 차량별 대기/이동/출발 상태 통계

---

## ⚙️ 설정

### 📹 웹캠 설정 (`webcam_blackbox_config.json`)

```json
{
  "camera": {
    "device_id": 0,        // USB 웹캠 번호 (0, 1, 2...)
    "width": 1280,         // 해상도 (높을수록 정밀, 낮을수록 빠름)  
    "height": 720,
    "fps": 30              // 프레임율 (높을수록 부드럽, 낮을수록 안정)
  },
  "detection": {
    "detection_interval": 1,      // N프레임마다 AI 분석 (1=매프레임, 2=2프레임마다)
    "confidence_threshold": 0.5   // 객체 감지 임계값
  }
}
```

### 📺 하이브리드 설정 (`hybrid_blackbox_config.json`)

```json
{
  "youtube": {
    "url": "your_default_url",
    "quality": "720p"       // 480p, 720p, 1080p, best
  },
  "tracking": {
    "movement_threshold": 1.5,    // 출발감지 민감도 (낮을수록 민감)
    "iou_threshold": 0.25         // 추적 정확도 (높을수록 정확)
  }
}
```

### 🎯 핵심 알고리즘 파라미터 (최적화 완료)

```python
출발감지_대기시간 = 1초      # 기존 2초 → 1초로 단축
정지상태_판정기준 = 2.5픽셀   # 기존 1.5px → 2.5px로 완화  
정지비율_임계값 = 50%        # 기존 70% → 50%로 완화
움직임감지_임계값 = 2.25픽셀  # 기존 4px → 2.25px로 민감화
```

### 🚦 신호등 HSV 색상 범위 (한국 신호등 최적화)

```python
빨간불: [0-10, 170-180] + [100-255, 100-255]
노란불: [18-35] + [120-255, 120-255]  
초록불: [45-90] + [100-255, 100-255]
```

### 🛣️ 차선 감지 알고리즘

- **Canny Edge Detection** → 차선 엣지 추출
- **Hough Transform** → 직선 패턴 감지  
- **동적 ROI** → 화면 하단 65% 영역 집중 분석
- **중심점 계산** → 좌우 차선 중심의 Front Zone 설정

---

## 🎛️ 키 조작법

### 🔧 기본 제어 키

| 키      | 기능       | 설명                  |
|---------|------------|-----------------------|
| **Q**   | 종료       | 프로그램 안전 종료    |
| **SPACE** | 일시정지   | 재생/일시정지 토글     |
| **S**   | 스크린샷   | 현재 화면 저장        |
| **ESC** | 긴급종료   | 강제 종료 (USB 연결 해제시) |
| **R**   | 재연결     | 웹캠 재연결 시도       |

### ⏯️ 시간 제어 키 (YouTube/MP4 모드)

| 키      | 기능       | 설명                  |
|---------|------------|-----------------------|
| **← →** | 점프       | 설정된 시간만큼 이동   |
| **↑ ↓** | 10초 점프  | 빠른 이동             |
| **A D** | 1초 미세조정 | 정밀 탐색            |
| **T**   | 점프시간 변경 | 5s → 10s → 30s → 60s... |

---

## 📊 성능 정보

### 💻 성능 벤치마크

| 해상도      | CPU 사용률 | 평균 FPS | 감지 지연   |
|-------------|------------|----------|-------------|
| **640x480** | ~30%      | 28-30   | <0.1초     |
| **1280x720**| ~45%      | 25-28   | <0.2초     |  
| **1920x1080**| ~65%     | 20-25   | <0.3초     |

### ⚡ 최적화 팁

**고성능 설정** (높은 정확도):
```json
{
  "camera": { "width": 1920, "height": 1080, "fps": 30 },
  "detection": { "detection_interval": 1 }
}
```

**경량 설정** (낮은 사양 PC):
```json
{
  "camera": { "width": 640, "height": 480, "fps": 15 },
  "detection": { "detection_interval": 3 }
}
```

---

## 🔧 문제해결

### ❌ 자주 발생하는 문제

#### 🚫 모델 파일 없음 오류
```bash
FileNotFoundError: yolov4-tiny.weights
```
**해결방법:**
```bash
wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
```

#### 📹 웹캠 연결 실패
```bash
cv2.error: Cannot open camera
```
**해결방법:**
```json
// webcam_blackbox_config.json에서 device_id 변경
{
  "camera": {
    "device_id": 1    // 0 → 1로 변경 후 재시도
  }
}
```

#### 🐌 성능 저하 문제
**해결방법:**
```json
// 성능 최적화 설정
{
  "camera": {
    "width": 640,     // 해상도 낮춤
    "height": 480,
    "fps": 15         // 목표 FPS 낮춤
  },
  "detection": {
    "detection_interval": 3  // AI 분석 주기 늘림
  }
}
```

#### 📺 YouTube 다운로드 오류
```bash
yt-dlp ERROR
```
**해결방법:**
```bash
pip install --upgrade yt-dlp
```

### 🚨 긴급 상황 대처

| 상황             | 해결 키 | 설명                |
|------------------|---------|---------------------|
| **프로그램 먹통** | `ESC`  | 강제 종료           |
| **USB 캠 연결 해제** | `Q`   | 안전 종료           |
| **카메라 재연결** | `R`    | 웹캠 재연결 시도    |

---

## 🛠️ 개발 정보

### 🔄 최근 업데이트 (v2.0, 2025년 기준)
- ✅ **강화된 USB 모니터링** 및 긴급종료 키 (`ESC`, `Q`) 추가
- ✅ **하이브리드 시스템**: YouTube + MP4 통합
- ✅ **출발감지 최적화**: 1초 대기, 50% 정지율
- ✅ **Downloads 폴더 통합**: 파일 관리 간소화

### 🔮 향후 계획
- 🚀 **GPU 가속**: CUDA/OpenCL 지원으로 실시간 4K 처리
- ☁️ **클라우드 연동**: AWS/GCP 실시간 스트리밍  
- 📱 **모바일 앱**: 스마트폰 원격 제어
- 🎥 **다중 카메라**: 전후방 동시 처리
- 🧠 **고급 AI**: 실시간 차선 변경 감지, 졸음운전 감지

### 🏆 개발 로드맵
- [ ] 실시간 차선 변경 감지
- [ ] 졸음운전 감지
- [ ] 스마트폰 미러링
- [ ] 블랙박스 영상 자동 편집

---

## 🤝 기여하기

이 프로젝트에 기여해주시는 모든 분들을 환영합니다!

### 📝 기여 방법
1. **Fork** the Project
2. **Create** Feature Branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** Changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to Branch (`git push origin feature/AmazingFeature`)
5. **Open** Pull Request

### 🐛 버그 리포트
Issues 탭에서 다음 정보와 함께 버그를 리포트해주세요:
- **OS**: Windows 10/11, Ubuntu 20.04, macOS 등
- **Python 버전**: 3.8, 3.9, 3.10 등
- **에러 메시지**: 전체 traceback
- **재현 단계**: 1-2-3 단계

---

## 📄 라이선스

이 프로젝트는 MIT License 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.

---

## 📞 연락처

- 📧 **이메일**: ojt8416@gmail.com
- 💬 **Issues**: [GitHub Issues](https://github.com/juntaek-oh/Blackbox/issues)
- 📖 **Wiki**: [프로젝트 위키](https://github.com/juntaek-oh/Blackbox/wiki)

---

<div align="center">

## 🚗 안전한 운전의 시작, AI 블랙박스 시스템과 함께하세요! 🚗

**AI 블랙박스 시스템 - 실시간 차량 추적의 새로운 패러다임**

🎯 **핵심 기술**: IoU 추적, 동적 차선 감지, HSV 신호등 인식  
🚀 **성능 최적화**: 1초 출발감지, 30fps 실시간 처리  
🎛️ **사용자 경험**: 화살표 키 시간조절, TTS 음성안내

---

⭐ **이 프로젝트가 도움이 되셨다면 Star를 눌러주세요!**  
🔄 **Pull Requests와 Issues를 환영합니다!**

**Made with ❤️ by AI 블랙박스 팀**

</div>
