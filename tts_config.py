# tts_config.py
# 블랙박스 시스템용 TTS 설정 (2025.08 버전)

import os

# 기본 디렉토리 설정
DEFAULT_AUDIO_DIR = "../audio"
DEFAULT_TEXT_DIR = "../log"
DEFAULT_TXT_FILENAME = "important.txt"

# 메시지 설정
MESSAGE_COOLDOWN = 3  # 초
MAX_WORKERS = 2

# 블랙박스용 메시지 매핑 (핵심 메시지만)
DETECTION2TEXT = {
    "신호가 변경되었습니다": {"log": "신호가 변경되었습니다", "priority": 1},
    "앞의 차량이 출발하였습니다": {"log": "앞의 차량이 출발하였습니다", "priority": 2},
    "차량이 감지되었습니다": {"log": "차량이 감지되었습니다", "priority": 3},
    "신호등이 감지되었습니다": {"log": "신호등이 감지되었습니다", "priority": 3}
}

# 오디오 플레이어 우선순위 (Windows 최적화)
AUDIO_PLAYERS = ['pygame', 'vlc', 'ffplay', 'mpg123', 'mpg321']

# TTS 설정
TTS_LANGUAGE = 'ko'
TTS_SLOW = False

# 로깅 설정
LOGGING_LEVEL = "INFO"
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# 파일 인코딩
FILE_ENCODING = "utf-8"

# 처리 간격
QUEUE_SLEEP_INTERVAL = 0.1
MESSAGE_INTERVAL = 0.5
