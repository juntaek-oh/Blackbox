# tts_settings.py
# 블랙박스 시스템용 TTS 모듈 (2025.08 버전)

import os
import threading
import time
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor

# Windows 환경 호환을 위한 추가 임포트
try:
    import pygame

    pygame.mixer.init()
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

try:
    from gtts import gTTS

    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False
    print("⚠️ gTTS가 설치되지 않았습니다. pip install gtts 실행하세요.")


class TTSNavigationSystem:
    def __init__(self, base_dir=None):
        # 기본 경로 설정 (__file__ 기준으로 변경)
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))

        self.base_dir = base_dir
        self.AUDIO_DIR = os.path.join(base_dir, "audio")
        self.TEXT_DIR = os.path.join(base_dir, "log")
        self.TXT_NAME = "important.txt"
        self.txt_path = os.path.join(self.TEXT_DIR, self.TXT_NAME)

        # 디렉토리 생성
        os.makedirs(self.AUDIO_DIR, exist_ok=True)
        os.makedirs(self.TEXT_DIR, exist_ok=True)

        # 텍스트 파일 초기화
        self._initialize_text_file()

        # 블랙박스용 메시지 매핑 (간소화)
        self.DETECTION2TEXT = {
            "신호가 변경되었습니다": {"log": "신호가 변경되었습니다", "priority": 1},
            "앞의 차량이 출발하였습니다": {"log": "앞의 차량이 출발하였습니다", "priority": 2},
            "차량이 감지되었습니다": {"log": "차량이 감지되었습니다", "priority": 3},
            "신호등이 감지되었습니다": {"log": "신호등이 감지되었습니다", "priority": 3}
        }

        # 설정값
        self.message_cooldown = 3  # 3초 쿨다운
        self.recent_messages = {}
        self.priority_queue = []
        self.queue_lock = threading.Lock()
        self.file_lock = threading.Lock()

        # 스레드풀
        self.executor = ThreadPoolExecutor(max_workers=2)

        # TTS 설정
        self.tts_language = 'ko'
        self.tts_slow = False

        logging.info("✅ TTS Navigation System 초기화 완료")

    def _initialize_text_file(self):
        """텍스트 파일 초기화"""
        try:
            if not os.path.exists(self.txt_path):
                with open(self.txt_path, "w", encoding='utf-8') as f:
                    f.write("")
                print(f"✓ TTS 텍스트 파일 생성: {self.txt_path}")
            else:
                print(f"✓ TTS 텍스트 파일 존재: {self.txt_path}")
        except Exception as e:
            print(f"✗ TTS 텍스트 파일 초기화 오류: {e}")
            logging.error(f"TTS 텍스트 파일 초기화 오류: {e}")

    def _read_text_lines(self):
        """텍스트 파일 읽기"""
        with self.file_lock:
            try:
                with open(self.txt_path, "r", encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                return lines
            except Exception as e:
                logging.error(f"파일 읽기 오류: {e}")
                return []

    def _add_text_to_file(self, text):
        """텍스트 파일에 추가"""
        with self.file_lock:
            try:
                with open(self.txt_path, "a", encoding='utf-8') as f:
                    f.write(text + "\n")
                logging.info(f"새 TTS 문구 추가: {text}")
            except Exception as e:
                logging.error(f"파일 쓰기 오류: {e}")

    def _find_or_add_text(self, text):
        """문구 찾기 또는 추가"""
        lines = self._read_text_lines()
        try:
            line_number = lines.index(text) + 1
            return line_number
        except ValueError:
            self._add_text_to_file(text)
            return len(lines) + 1

    def _generate_mp3_filename(self, keyword):
        """MP3 파일명 생성"""
        safe_keyword = "".join(c if c.isalnum() or c in '-_' else '_' for c in str(keyword))
        return f"tts_{safe_keyword}.mp3"

    def _generate_audio_file(self, text, mp3_path):
        """TTS 오디오 파일 생성"""
        if not HAS_GTTS:
            print("✗ gTTS가 설치되지 않았습니다.")
            return False

        try:
            if not os.path.exists(mp3_path):
                tts = gTTS(text=text, lang=self.tts_language, slow=self.tts_slow)
                tts.save(mp3_path)
                print(f"✓ TTS 파일 생성: {os.path.basename(mp3_path)}")
                logging.info(f"TTS 파일 생성: {mp3_path}")
            return True
        except Exception as e:
            print(f"✗ TTS 생성 실패: {e}")
            logging.error(f"TTS 생성 오류: {e}")
            return False

    def _play_audio_pygame(self, mp3_path):
        """pygame으로 오디오 재생 (Windows 최적화)"""
        try:
            if not HAS_PYGAME:
                return False

            pygame.mixer.music.load(mp3_path)
            pygame.mixer.music.play()

            # 재생 완료까지 대기
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            print(f"✓ pygame으로 재생 완료: {os.path.basename(mp3_path)}")
            return True
        except Exception as e:
            print(f"✗ pygame 재생 실패: {e}")
            return False

    def _play_audio_system(self, mp3_path):
        """시스템 명령어로 오디오 재생"""
        players = ['vlc', 'ffplay', 'mpg123', 'mpg321']

        for player in players:
            try:
                # Windows에서 which 대신 where 사용
                check_cmd = f"where {player}" if os.name == 'nt' else f"which {player}"
                if os.system(f"{check_cmd} >nul 2>&1" if os.name == 'nt' else f"{check_cmd} >/dev/null 2>&1") == 0:
                    play_cmd = f'{player} "{mp3_path}"'
                    if os.name == 'nt':
                        play_cmd += " >nul 2>&1"
                    else:
                        play_cmd += " >/dev/null 2>&1"

                    if os.system(play_cmd) == 0:
                        print(f"✓ {player}로 재생 완료: {os.path.basename(mp3_path)}")
                        return True
            except Exception as e:
                continue

        return False

    def _play_audio(self, mp3_path):
        """오디오 재생 (pygame 우선, 시스템 명령어 백업)"""
        if not os.path.exists(mp3_path):
            print(f"✗ 파일이 존재하지 않음: {mp3_path}")
            return False

        # pygame 우선 시도 (Windows 환경에서 가장 안정적)
        if self._play_audio_pygame(mp3_path):
            return True

        # 시스템 명령어 백업
        if self._play_audio_system(mp3_path):
            return True

        print("✗ 사용 가능한 오디오 플레이어를 찾을 수 없습니다.")
        print("  다음 플레이어 중 하나를 설치해주세요: pygame, vlc, ffmpeg")
        return False

    def is_recently_played(self, text):
        """최근 재생 확인 (중복 방지)"""
        current_time = time.time()
        if text in self.recent_messages:
            if current_time - self.recent_messages[text] < self.message_cooldown:
                return True
        self.recent_messages[text] = current_time
        return False

    def play_situation_from_txt(self, situation_text, keyword=None):
        """상황별 TTS 재생 (블랙박스 메인 함수)"""
        try:
            # 중복 재생 방지
            if self.is_recently_played(situation_text):
                return False

            # 키워드 기반 파일명 생성
            if keyword:
                mp3_filename = self._generate_mp3_filename(keyword)
            else:
                mp3_filename = self._generate_mp3_filename(situation_text)

            mp3_path = os.path.join(self.AUDIO_DIR, mp3_filename)

            # MP3 파일 생성 (없으면)
            if self._generate_audio_file(situation_text, mp3_path):
                return self._play_audio(mp3_path)
            else:
                return False

        except Exception as e:
            print(f"✗ TTS 재생 처리 오류: {e}")
            logging.error(f"TTS 재생 처리 오류: {e}")
            return False

    def announce_detection(self, detected_text, keyword=None, force_play=False):
        """탐지 결과 TTS 안내 (블랙박스 호출용)"""
        try:
            if force_play or not self.is_recently_played(detected_text):
                return self.play_situation_from_txt(detected_text, keyword)
            return False
        except Exception as e:
            logging.error(f"TTS 안내 오류: {e}")
            return False

    def test_tts_basic(self):
        """기본 TTS 테스트"""
        test_message = "TTS 시스템 테스트입니다."
        print("🔊 TTS 기본 테스트 시작...")
        result = self.play_situation_from_txt(test_message, "test")
        if result:
            print("✅ TTS 테스트 성공")
        else:
            print("❌ TTS 테스트 실패")
        return result

    def shutdown(self):
        """TTS 시스템 종료"""
        try:
            self.executor.shutdown(wait=True)
            if HAS_PYGAME:
                pygame.mixer.quit()
            logging.info("TTS Navigation System 종료")
        except Exception as e:
            logging.error(f"TTS 종료 오류: {e}")

    def show_status(self):
        """TTS 시스템 상태 표시"""
        print(f"\n=== TTS 시스템 상태 ===")
        print(f"오디오 디렉토리: {self.AUDIO_DIR}")
        print(f"텍스트 파일: {self.txt_path}")
        print(f"pygame 사용 가능: {HAS_PYGAME}")
        print(f"gTTS 사용 가능: {HAS_GTTS}")

        # MP3 파일 목록
        if os.path.exists(self.AUDIO_DIR):
            mp3_files = [f for f in os.listdir(self.AUDIO_DIR) if f.endswith('.mp3')]
            print(f"생성된 MP3 파일: {len(mp3_files)}개")
            for mp3 in mp3_files:
                print(f"  • {mp3}")
        print("=" * 30)
