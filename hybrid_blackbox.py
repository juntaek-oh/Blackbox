#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
하이브리드 블랙박스 시스템 v4.4 - 신호등 오탐지 해결 + mp4_output 저장 추가
YouTube URL + Downloads 폴더 통합 + 모드별 최적화 UI + 빠른 출발 감지
"""

import cv2
import numpy as np
import time
import json
import logging
import os
import signal
import sys
import psutil
from datetime import datetime, timedelta
from collections import deque
import argparse
import queue
import threading
import hashlib
import math
import requests
import subprocess
import tempfile

# YouTube 처리를 위한 라이브러리
try:
    import yt_dlp

    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    print("⚠️ yt-dlp 라이브러리가 설치되지 않았습니다. 'pip install yt-dlp' 실행 후 재시도하세요.")

from tts_config import LOGGING_LEVEL, LOGGING_FORMAT
from tts_settings import TTSNavigationSystem

# 전역 로깅 설정
logging.basicConfig(level=getattr(logging, LOGGING_LEVEL), format=LOGGING_FORMAT)


class HybridVideoManager:
    """하이브리드 비디오 관리자 - YouTube 스트림 + MP4 파일 통합"""

    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.video_capture = None
        self.video_fps = 30
        self.total_frames = 0
        self.current_frame_idx = 0
        self.video_duration = 0

        # 모드 관련
        self.mode = None  # 'stream' or 'file'
        self.youtube_url = None
        self.video_file_path = None
        self.temp_file_path = None
        self.stream_url = None
        self.video_info = {}

        # 크기 관리
        self.original_width = 1280
        self.original_height = 720
        self.display_width = 1280
        self.display_height = 720

        # ✅ mp4_output 저장 관련 추가
        self.output_dir = "mp4_output"
        self.video_writer = None
        self.recording_enabled = True
        self.create_output_directory()

    def create_output_directory(self):
        """mp4_output 디렉터리 생성"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.logger.info(f"✅ 출력 디렉터리 생성: {os.path.abspath(self.output_dir)}")
        except Exception as e:
            self.logger.error(f"출력 디렉터리 생성 실패: {e}")

    def init_video_writer(self, frame_shape):
        """VideoWriter 초기화 (mp4_output 폴더에 저장)"""
        if not self.recording_enabled:
            return True

        try:
            # 현재 시간으로 파일명 생성
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = os.path.join(self.output_dir, f"hybrid_blackbox_{current_time}.mp4")

            height, width = frame_shape[:2]
            fps = self.config.get('video_mode', {}).get('fps', 30) or 30

            # VideoWriter 생성
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # 크기가 홀수인 경우 짝수로 맞춤 (codec 호환성)
            if width % 2 != 0:
                width -= 1
            if height % 2 != 0:
                height -= 1

            self.video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

            if not self.video_writer.isOpened():
                self.logger.error("❌ VideoWriter 초기화 실패")
                return False

            self.logger.info(f"✅ 영상 저장 시작: {os.path.basename(output_filename)} ({width}x{height}@{fps}fps)")
            return True

        except Exception as e:
            self.logger.error(f"VideoWriter 초기화 실패: {e}")
            return False

    def write_frame(self, frame):
        """프레임을 mp4_output에 저장"""
        if not self.recording_enabled or not self.video_writer:
            return

        try:
            # 프레임 크기를 짝수로 맞춤
            h, w = frame.shape[:2]
            if w % 2 != 0:
                frame = frame[:, :-1]
                w -= 1
            if h % 2 != 0:
                frame = frame[:-1, :]
                h -= 1

            self.video_writer.write(frame)
        except Exception as e:
            self.logger.error(f"프레임 저장 실패: {e}")

    def init_from_youtube_url(self, youtube_url, mode='stream', quality='720p'):
        """YouTube URL로부터 초기화 (스트림 모드 또는 다운로드 모드)"""
        if not YT_DLP_AVAILABLE:
            self.logger.error("❌ yt-dlp library required.")
            return False

        self.youtube_url = youtube_url
        self.mode = mode

        if mode == 'stream':
            return self._init_stream_mode()
        elif mode == 'download':
            return self._init_download_mode(quality)
        else:
            self.logger.error(f"❌ Unknown mode: {mode}")
            return False

    def init_from_file(self, file_path):
        """MP4 파일로부터 직접 초기화"""
        self.mode = 'file'
        self.video_file_path = file_path
        return self._init_file_mode()

    def _init_stream_mode(self):
        """YouTube 스트림 모드 초기화"""
        self.logger.info(f"📡 YouTube 스트림 모드로 초기화: {self.youtube_url}")

        try:
            # YouTube 스트림 URL 추출
            quality_formats = [
                'best[height<=1080][ext=mp4]/best[height<=1080]',
                'best[height<=720][ext=mp4]/best[height<=720]',
                'best[height<=480][ext=mp4]/best[height<=480]',
                'best[ext=mp4]/best'
            ]

            for format_selector in quality_formats:
                try:
                    ydl_opts = {
                        'format': format_selector,
                        'quiet': True,
                        'no_warnings': True,
                        'extract_flat': False,
                    }

                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        self.video_info = ydl.extract_info(self.youtube_url, download=False)

                        if 'url' in self.video_info:
                            self.stream_url = self.video_info['url']
                            break
                        elif 'formats' in self.video_info:
                            for fmt in self.video_info['formats']:
                                if (fmt.get('vcodec') != 'none' and
                                        fmt.get('acodec') != 'none' and
                                        'url' in fmt):
                                    self.stream_url = fmt['url']
                                    break

                        if self.stream_url:
                            break

                except Exception as e:
                    continue

            if not self.stream_url:
                self.logger.error("❌ 스트림 URL을 찾을 수 없습니다.")
                return False

            # 스트림 연결
            cap = cv2.VideoCapture(self.stream_url)
            if not cap.isOpened():
                self.logger.error("❌ 스트림 연결 실패")
                return False

            # 첫 프레임으로 정보 확인
            ret, frame = cap.read()
            if not ret or frame is None:
                self.logger.error("❌ 스트림에서 프레임을 읽을 수 없습니다.")
                cap.release()
                return False

            self._setup_video_info(cap, frame)
            self.video_capture = cap

            # ✅ VideoWriter 초기화
            if not self.init_video_writer(frame.shape):
                self.logger.warning("VideoWriter 초기화 실패, 저장 없이 진행")

            self.logger.info("✅ YouTube 스트림 모드 초기화 완료")
            return True

        except Exception as e:
            self.logger.error(f"스트림 모드 초기화 실패: {e}")
            return False

    def _init_download_mode(self, quality='720p'):
        """YouTube 다운로드 후 MP4 모드 초기화"""
        self.logger.info(f"📥 YouTube 다운로드 모드로 초기화: {self.youtube_url}")

        try:
            # 현재 실행 위치에 downloads 폴더 생성
            current_dir = os.getcwd()
            download_dir = os.path.join(current_dir, "downloads")
            os.makedirs(download_dir, exist_ok=True)

            # YouTube 제목으로 파일명 생성
            print("📋 YouTube 영상 정보를 가져오는 중...")
            try:
                with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                    self.video_info = ydl.extract_info(self.youtube_url, download=False)
                    title = self.video_info.get('title', 'youtube_video')
                    # 파일명으로 사용할 수 없는 문자 제거
                    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    safe_title = safe_title[:50]  # 길이 제한
            except:
                safe_title = "youtube_video"

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_title}_{timestamp}.mp4"
            self.temp_file_path = os.path.join(download_dir, filename)

            quality_map = {
                '1080p': 'best[height<=1080][ext=mp4]/best[height<=1080]',
                '720p': 'best[height<=720][ext=mp4]/best[height<=720]',
                '480p': 'best[height<=480][ext=mp4]/best[height<=480]',
                'best': 'best[ext=mp4]/best'
            }
            format_selector = quality_map.get(quality, quality_map['720p'])

            def progress_hook(d):
                if d['status'] == 'downloading':
                    try:
                        if self._check_download_interrupt():
                            raise KeyboardInterrupt("사용자가 다운로드를 중단했습니다.")

                        if 'total_bytes' in d:
                            downloaded = d.get('downloaded_bytes', 0)
                            total = d['total_bytes']
                            percent = (downloaded / total) * 100
                            print(
                                f"\r📥 다운로드 진행률: {percent:.1f}% ({downloaded // 1024 // 1024}MB/{total // 1024 // 1024}MB) [Ctrl+C로 중단]",
                                end="")
                        elif '_percent_str' in d:
                            percent_str = d['_percent_str']
                            print(f"\r📥 다운로드 진행률: {percent_str} [Ctrl+C로 중단]", end="")
                    except KeyboardInterrupt:
                        print(f"\n⚠️ 사용자가 다운로드를 중단했습니다.")
                        raise
                elif d['status'] == 'finished':
                    print(f"\n✅ 다운로드 완료!")

            ydl_opts = {
                'format': format_selector,
                'outtmpl': self.temp_file_path,
                'quiet': False,
                'no_warnings': True,
                'progress_hooks': [progress_hook],
            }

            print("\n" + "=" * 70)
            print(f"📥 YouTube 영상 다운로드를 시작합니다")
            print(f"📁 다운로드 위치: {os.path.abspath(download_dir)}")
            print(f"📄 파일명: {filename}")
            print(f"🎞️ 품질: {quality}")
            print("💡 중단하려면 Ctrl+C를 누르세요.")
            print("=" * 70)

            try:
                duration = self.video_info.get('duration', 0)
                print(
                    f"📹 제목: {self.video_info.get('title', '')[:60]}{'...' if len(self.video_info.get('title', '')) > 60 else ''}")
                print(f"⏱️ 길이: {duration // 60:02d}:{duration % 60:02d}")
                print("📥 다운로드 시작...")

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([self.youtube_url])

            except KeyboardInterrupt:
                print(f"\n🛑 다운로드가 중단되었습니다.")
                self._cleanup_temp_files()
                choice = input("🔄 스트림 모드로 전환하시겠습니까? (y/n): ").lower()
                if choice in ['y', 'yes', '예', '']:
                    print("📡 스트림 모드로 전환합니다...")
                    return self._init_stream_mode()
                else:
                    return False

            if not os.path.exists(self.temp_file_path):
                self.logger.error("❌ 다운로드된 파일을 찾을 수 없습니다.")
                return False

            # 다운로드 완료 후 정보 표시
            file_size = os.path.getsize(self.temp_file_path) / (1024 * 1024)
            print("\n" + "🎉" + "=" * 68 + "🎉")
            print("✅ 다운로드 완료! 이제 MP4 모드로 실행합니다.")
            print(f"📁 저장 위치: {os.path.abspath(self.temp_file_path)}")
            print(f"📊 파일 크기: {file_size:.1f}MB")
            print("🎛️ 화살표 키로 자유롭게 시간 이동이 가능합니다!")
            print("🎉" + "=" * 68 + "🎉")

            # 다운로드된 파일을 재생 파일로 설정
            self.video_file_path = self.temp_file_path
            self.mode = 'file'  # 모드를 file로 설정하여 시간 조절 활성화
            return self._init_file_mode()

        except Exception as e:
            self.logger.error(f"다운로드 모드 초기화 실패: {e}")
            self._cleanup_temp_files()
            return False

    def _init_file_mode(self):
        """MP4 파일 모드 초기화"""
        if not os.path.exists(self.video_file_path):
            self.logger.error(f"❌ 파일을 찾을 수 없습니다: {self.video_file_path}")
            return False

        self.logger.info(f"📹 MP4 파일 모드로 초기화: {self.video_file_path}")

        cap = cv2.VideoCapture(self.video_file_path)
        if not cap.isOpened():
            self.logger.error("❌ MP4 파일을 열 수 없습니다.")
            return False

        ret, frame = cap.read()
        if not ret or frame is None:
            self.logger.error("❌ MP4 파일에서 프레임을 읽을 수 없습니다.")
            cap.release()
            return False

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._setup_video_info(cap, frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.video_capture = cap
        self.current_frame_idx = 0

        # ✅ VideoWriter 초기화
        if not self.init_video_writer(frame.shape):
            self.logger.warning("VideoWriter 초기화 실패, 저장 없이 진행")

        self.logger.info("✅ MP4 파일 모드 초기화 완료")
        self.logger.info("✅ 완전한 시간 조절 기능 사용 가능")
        self.logger.info(f"▶️ 재생 중: {os.path.basename(self.video_file_path)}")
        return True

    def _setup_video_info(self, cap, frame):
        """비디오 정보 설정"""
        h, w = frame.shape[:2]
        self.original_width = w
        self.original_height = h
        self.display_width = w
        self.display_height = h

        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        if self.video_fps <= 0 or self.video_fps > 60:
            self.video_fps = 30

        if self.mode == 'file':
            self.video_duration = self.total_frames / self.video_fps
        else:
            self.video_duration = self.video_info.get('duration', 0)
            self.total_frames = int(self.video_duration * self.video_fps) if self.video_duration > 0 else 999999

        self.logger.info(f"✅ 비디오 정보: {w}x{h}, {self.video_fps:.1f}fps, {self.video_duration:.1f}초")

    def _check_download_interrupt(self):
        """다운로드 중 키 입력 체크 (논블로킹)"""
        try:
            if os.name == 'nt':
                import msvcrt
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b'\x03' or key == b'\x1b':
                        return True
            else:
                import select
                import sys
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    if ord(key) in [3, 27]:
                        return True
        except:
            pass
        return False

    def _cleanup_temp_files(self):
        """임시 파일 정리"""
        try:
            if hasattr(self, 'temp_file_path') and self.temp_file_path:
                if os.path.exists(self.temp_file_path):
                    download_dir = os.path.join(os.getcwd(), "downloads")
                    if os.path.dirname(self.temp_file_path) == download_dir:
                        self.logger.info(f"💾 다운로드된 파일 보관: {self.temp_file_path}")
                    else:
                        os.remove(self.temp_file_path)
                        self.logger.info("🧹 임시 파일 정리됨")
        except Exception as e:
            self.logger.warning(f"임시 파일 정리 실패: {e}")

    def jump_to_time(self, target_seconds):
        """시간 점프 (파일 모드에서만)"""
        if self.mode != 'file':
            return False

        if not self.video_capture or not self.video_capture.isOpened():
            return False

        target_frame = int(target_seconds * self.video_fps)
        target_frame = max(0, min(target_frame, self.total_frames - 1))

        try:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            self.current_frame_idx = target_frame
            current_time = self.get_current_time()
            self.logger.info(f"⭐ 시간 이동: {current_time:.1f}초")
            return True
        except Exception as e:
            return False

    def jump_relative(self, seconds_delta):
        """상대적 시간 이동"""
        if self.mode != 'file':
            return False

        current_time = self.get_current_time()
        target_time = current_time + seconds_delta
        target_time = max(0, min(target_time, self.video_duration))

        return self.jump_to_time(target_time)

    def get_current_time(self):
        """현재 재생 시간 반환"""
        if self.video_fps == 0:
            return 0
        return self.current_frame_idx / self.video_fps

    def get_total_duration(self):
        """전체 길이 반환"""
        return self.video_duration

    def read_frame(self):
        """프레임 읽기"""
        if not self.video_capture or not self.video_capture.isOpened():
            return False, None

        ret, frame = self.video_capture.read()

        if ret and frame is not None:
            self.current_frame_idx += 1
            return True, frame
        else:
            if self.mode == 'file':
                loop_enabled = self.config.get('video_mode', {}).get('loop', True)
                if loop_enabled and self.total_frames > 0:
                    self.logger.info("🔄 비디오 반복 재생 시작")
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame_idx = 0
                    ret, frame = self.video_capture.read()
                    if ret and frame is not None:
                        self.current_frame_idx = 1
                        return True, frame
            else:
                self.logger.warning("⚠️ 스트림 연결이 끊어졌습니다.")

            return False, None

    def get_fps(self):
        return self.video_fps

    def get_progress(self):
        if self.total_frames == 0:
            return 0.0
        return min(self.current_frame_idx / self.total_frames, 1.0)

    def get_mode_info(self):
        """현재 모드 정보 반환"""
        return {
            'mode': self.mode,
            'youtube_url': self.youtube_url,
            'file_path': self.video_file_path,
            'can_seek': self.mode == 'file',  # 파일 모드에서만 시간 조절 가능
            'title': self.video_info.get('title', 'Unknown') if self.video_info else 'Unknown'
        }

    def release(self):
        """리소스 해제"""
        # ✅ VideoWriter 해제 추가
        if self.video_writer:
            self.video_writer.release()
            self.logger.info("✅ 영상 저장 완료")

        if self.video_capture:
            self.video_capture.release()

        if hasattr(self, 'temp_file_path') and self.temp_file_path:
            self._cleanup_temp_files()


class TimeNavigator:
    """시간 탐색 기능 클래스 (파일 모드 전용)"""

    def __init__(self, logger):
        self.logger = logger
        self.jump_seconds = [5, 10, 30, 60, 120, 300]
        self.current_jump_index = 2  # 기본 30초
        self.show_navigation_ui = False
        self.ui_show_time = 0
        self.ui_display_duration = 3.0

    def get_current_jump_seconds(self):
        return self.jump_seconds[self.current_jump_index]

    def cycle_jump_time(self):
        self.current_jump_index = (self.current_jump_index + 1) % len(self.jump_seconds)
        self.show_navigation_ui = True
        self.ui_show_time = time.time()
        jump_time = self.get_current_jump_seconds()
        self.logger.info(f"⏱️ 점프 시간 설정: {jump_time}초")
        return jump_time

    def should_show_ui(self):
        if not self.show_navigation_ui:
            return False

        if time.time() - self.ui_show_time > self.ui_display_duration:
            self.show_navigation_ui = False
            return False

        return True

    def format_time(self, seconds):
        if seconds < 0:
            return "00:00"

        hours = int(seconds // 3600)
        minutes = int(seconds % 3600) // 60
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"


class LaneDetector:
    """차선 감지"""

    def __init__(self):
        self.lane_history = deque(maxlen=5)
        self.last_valid_lanes = None

    def detect_lanes(self, frame):
        height, width = frame.shape[:2]
        roi_vertices = np.array([[
            (0, height), (int(width * 0.5) - 50, int(height * 0.65)),
            (int(width * 0.5) + 50, int(height * 0.65)), (width, height)
        ]], dtype=np.int32)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi / 180,
                                threshold=50, minLineLength=50, maxLineGap=50)

        return self.process_lane_lines(lines, width, height)

    def process_lane_lines(self, lines, width, height):
        if lines is None:
            return self.last_valid_lanes

        left_lines, right_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if slope < -0.5:
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.5:
                right_lines.append((x1, y1, x2, y2))

        left_lane = self.average_lane(left_lines, width, height)
        right_lane = self.average_lane(right_lines, width, height)

        if left_lane is not None or right_lane is not None:
            lanes = {'left': left_lane, 'right': right_lane}
            self.lane_history.append(lanes)
            self.last_valid_lanes = lanes
            return lanes

        return self.last_valid_lanes

    def average_lane(self, lane_lines, width, height):
        if not lane_lines:
            return None

        x_coords, y_coords = [], []
        for x1, y1, x2, y2 in lane_lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])

        if len(x_coords) < 2:
            return None

        poly = np.polyfit(y_coords, x_coords, 1)
        y1 = height
        y2 = int(height * 0.6)
        x1 = int(poly[0] * y1 + poly[1])
        x2 = int(poly[0] * y2 + poly[1])

        return [x1, y1, x2, y2]

    def calculate_lane_center_points(self, lanes, height):
        if not lanes or (lanes.get('left') is None and lanes.get('right') is None):
            return None

        center_points = []
        for y in range(int(height * 0.65), height, 20):
            left_x, right_x = None, None

            if lanes.get('left'):
                x1, y1, x2, y2 = lanes['left']
                if y2 != y1:
                    left_x = x1 + (x2 - x1) * (y - y1) / (y2 - y1)

            if lanes.get('right'):
                x1, y1, x2, y2 = lanes['right']
                if y2 != y1:
                    right_x = x1 + (x2 - x1) * (y - y1) / (y2 - y1)

            if left_x is not None and right_x is not None:
                center_x = (left_x + right_x) / 2
            elif left_x is not None:
                center_x = left_x + 60
            elif right_x is not None:
                center_x = right_x - 60
            else:
                continue

            center_points.append((int(center_x), y))

        return center_points


class ImprovedIOUTracker:
    """개선된 IoU 추적기 - 더 민감한 출발 감지"""

    def __init__(self, max_lost=5, iou_threshold=0.3, movement_threshold=1.5, expected_fps=30):
        self.tracks = {}
        self.next_id = 1
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.movement_threshold = movement_threshold
        self.frame_count = 0
        self.expected_fps = expected_fps

        # ✅ 더 빠른 반응을 위한 완화된 설정
        self.departure_buffer_frames = int(expected_fps * 1.0)  # 2초 → 1초로 단축
        self.stationary_threshold = 2.5  # 1.5 → 2.5픽셀로 완화
        self.min_stationary_ratio = 0.5  # 70% → 50%로 완화

    @staticmethod
    def calculate_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def detect_departure_improved(self, track, is_in_zone=False):
        if len(track['movement_history']) < self.departure_buffer_frames:
            return False

        # ✅ 더 짧은 분석 구간
        recent_movements = list(track['movement_history'])[-self.departure_buffer_frames:]
        analysis_frames = recent_movements[:-8] if len(recent_movements) > 8 else recent_movements[:-3]

        if not analysis_frames:
            return False

        stationary_count = 0
        for move_x, move_y in analysis_frames:
            distance = (move_x ** 2 + move_y ** 2) ** 0.5
            if distance < self.stationary_threshold:  # ✅ 2.5픽셀로 완화
                stationary_count += 1

        stationary_ratio = stationary_count / len(analysis_frames)
        if stationary_ratio < self.min_stationary_ratio:  # ✅ 50%로 완화
            return False

        # ✅ 더 짧은 최종 움직임 분석
        recent_movement_frames = recent_movements[-5:] if len(recent_movements) >= 5 else recent_movements[-3:]
        avg_recent_distance = sum(
            (move_x ** 2 + move_y ** 2) ** 0.5
            for move_x, move_y in recent_movement_frames
        ) / len(recent_movement_frames)

        # ✅ 더 낮은 임계값 (1.5 * 1.5 = 2.25픽셀 vs 기존 4픽셀)
        adjusted_threshold = self.movement_threshold * (1.5 if is_in_zone else 2.0)
        return avg_recent_distance > adjusted_threshold

    def update(self, detections):
        self.frame_count += 1

        active_tracks = {tid: tr for tid, tr in self.tracks.items() if tr['lost'] <= self.max_lost}

        matched_tracks, matched_dets = set(), set()

        for track_id, track in active_tracks.items():
            best_iou, best_idx = 0.0, -1

            for i, det in enumerate(detections):
                if i in matched_dets:
                    continue

                iou = self.calculate_iou(track['bbox'], det['box'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou, best_idx = iou, i

            if best_idx != -1:
                old_bbox = track['bbox']
                new_bbox = detections[best_idx]['box']

                old_cx = old_bbox[0] + old_bbox[2] / 2
                old_cy = old_bbox[1] + old_bbox[3] / 2
                new_cx = new_bbox[0] + new_bbox[2] / 2
                new_cy = new_bbox[1] + new_bbox[3] / 2

                movement_x = new_cx - old_cx
                movement_y = new_cy - old_cy

                track['bbox'] = new_bbox
                track['confidence'] = detections[best_idx]['confidence']
                track['class_name'] = detections[best_idx]['class_name']
                track['lost'] = 0
                track['movement_history'].append((movement_x, movement_y))
                track['last_update'] = self.frame_count

                is_in_zone = detections[best_idx].get('in_zone', False)
                if self.detect_departure_improved(track, is_in_zone):
                    track['is_moving'] = True
                    track['departure_detected'] = True

                matched_tracks.add(track_id)
                matched_dets.add(best_idx)

        # 매칭되지 않은 기존 트랙 처리
        for track_id, track in active_tracks.items():
            if track_id not in matched_tracks:
                track['lost'] += 1

        # 새 트랙 생성
        max_history = max(self.departure_buffer_frames + 20, 100)
        for i, det in enumerate(detections):
            if i not in matched_dets:
                self.tracks[self.next_id] = {
                    'id': self.next_id,
                    'bbox': det['box'],
                    'confidence': det['confidence'],
                    'class_name': det['class_name'],
                    'lost': 0,
                    'created_frame': self.frame_count,
                    'last_update': self.frame_count,
                    'movement_history': deque(maxlen=max_history),
                    'is_moving': False,
                    'departure_detected': False
                }
                self.next_id += 1

        self.tracks = {tid: tr for tid, tr in self.tracks.items() if tr['lost'] <= self.max_lost}

        return self.get_active_tracks()

    def get_active_tracks(self):
        return [tr for tr in self.tracks.values() if tr['lost'] == 0]


class TrafficLightColorDetector:
    """신호등 색상 감지기"""

    def __init__(self, stability_frames=2):
        self.stability_frames = stability_frames
        self.color_history = deque(maxlen=stability_frames)
        self.last_stable_color = None
        self.last_change_time = 0
        self.previous_color = None
        self.confidence_threshold = 0.5

    def detect_traffic_light_color(self, roi):
        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 100, 100])
        red_upper2 = np.array([180, 255, 255])

        yellow_lower = np.array([18, 120, 120])
        yellow_upper = np.array([35, 255, 255])

        green_lower = np.array([45, 100, 100])
        green_upper = np.array([90, 255, 255])

        red_mask = cv2.inRange(hsv, red_lower1, red_upper1) + cv2.inRange(hsv, red_lower2, red_upper2)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        green_pixels = cv2.countNonZero(green_mask)

        max_pixels = max(red_pixels, yellow_pixels, green_pixels)

        if max_pixels < 15:
            return None

        if red_pixels == max_pixels:
            return 'red'
        elif yellow_pixels == max_pixels:
            return 'yellow'
        elif green_pixels == max_pixels:
            return 'green'

        return None

    def update_color_history(self, color):
        if color:
            self.color_history.append(color)

            if len(self.color_history) >= self.stability_frames:
                counts = {}
                for c in self.color_history:
                    counts[c] = counts.get(c, 0) + 1

                most_common = max(counts, key=counts.get)
                confidence = counts[most_common] / self.stability_frames

                if confidence >= self.confidence_threshold:
                    if self.last_stable_color != most_common:
                        change_info = {
                            'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'timestamp': time.time()
                        }

                        self.previous_color = self.last_stable_color
                        self.last_stable_color = most_common
                        self.last_change_time = time.time()
                        return change_info

        return None


class HybridBlackBoxSystem:
    """하이브리드 블랙박스 시스템 - 최적화된 UI"""

    def __init__(self, config_path="hybrid_blackbox_config.json"):
        self.setup_logging()
        self.config_path = config_path
        self.config = self.load_config()

        self.video_manager = HybridVideoManager(self.logger, self.config)
        self.time_navigator = TimeNavigator(self.logger)

        self.net = None
        self.classes = []
        self.output_layers = []
        self.running = False
        self.current_frame = None
        self.paused = False

        expected_fps = 30
        self.vehicle_tracker = ImprovedIOUTracker(
            max_lost=int(self.config.get('tracking', {}).get('max_lost_frames', 8)),
            iou_threshold=float(self.config.get('tracking', {}).get('iou_threshold', 0.25)),
            movement_threshold=float(self.config.get('tracking', {}).get('movement_threshold', 1.5)),
            expected_fps=expected_fps
        )

        self.traffic_light_detector = TrafficLightColorDetector(
            stability_frames=int(self.config.get('traffic_light', {}).get('stability_frames', 2))
        )

        self.lane_detector = LaneDetector()

        self.detection_stats = {
            'total_detections': 0,
            'vehicles': 0,
            'traffic_lights': 0,
            'traffic_light_changes': 0,
            'vehicle_departures': 0
        }

        self.processing_times = deque(maxlen=30)

        self.logger.info("✅ 하이브리드 블랙박스 시스템 초기화 완료")

        # TTS 시스템 초기화
        try:
            self.tts_system = TTSNavigationSystem()
            self.TTS_ALLOWED = ["신호가 변경되었습니다", "앞의 차량이 출발하였습니다"]
            self.tts_message_hashes = set()
        except Exception as e:
            self.logger.warning(f"TTS 시스템 초기화 실패: {e}")
            self.tts_system = None

    def setup_logging(self):
        os.makedirs("log", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('log/hybrid_blackbox.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self):
        default_config = {
            "hybrid": {
                "default_mode": "ask",
                "default_quality": "720p",
                "auto_delete_temp": True
            },
            "youtube": {
                "url": "https://www.youtube.com/watch?v=OoHzud9L48s",
                "output_dir": "output"
            },
            "video_mode": {
                "loop": True,
                "speed": 1.0,
                "save_result": True,
                "fps": 30  # ✅ VideoWriter용 FPS 설정 추가
            },
            "model": {
                "weights_path": "yolov4-tiny.weights",
                "config_path": "yolov4-tiny.cfg",
                "classes_path": "coco.names",
                "input_size": 416
            },
            "detection": {
                "confidence_threshold": 0.5,
                "nms_threshold": 0.4,
                "traffic_light_confidence_threshold": 0.3,
                "detection_interval": 2
            },
            "traffic_light": {
                "enable_detection": True,
                "stability_frames": 2
            },
            "tracking": {
                "iou_threshold": 0.25,
                "max_lost_frames": 8,
                "movement_threshold": 1.5
            },
            "display": {
                "show_preview": True,
                "window_width": 1280,
                "window_height": 720
            }
        }

        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)

                def merge(a, b):
                    for k, v in b.items():
                        if k not in a:
                            a[k] = v
                        elif isinstance(v, dict):
                            merge(a[k], v)
                    return a

                return merge(cfg, default_config)
            else:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
                return default_config
        except Exception as e:
            self.logger.error(f"설정 파일 로드 실패: {e}")
            return default_config

    def show_downloads_folder_info(self):
        """downloads 폴더 정보 표시"""
        download_dir = os.path.join(os.getcwd(), "downloads")
        if os.path.exists(download_dir):
            files = [f for f in os.listdir(download_dir) if f.endswith('.mp4')]
            if files:
                print(f"\n📁 Downloads 폴더: {os.path.abspath(download_dir)}")
                print(f"📹 저장된 영상: {len(files)}개")
                for i, file in enumerate(files[-3:], 1):
                    file_path = os.path.join(download_dir, file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"   {i}) {file[:40]}{'...' if len(file) > 40 else ''} ({file_size:.1f}MB)")

    def get_user_choice(self):
        """✅ 간소화된 사용자 선택 (MP4 직접 선택 제거)"""
        print("\n" + "=" * 80)
        print("   🎬 하이브리드 블랙박스 시스템 v4.4")
        print("   YouTube URL + Downloads 폴더 통합 관리 + mp4_output 저장")
        print("=" * 80)

        self.show_downloads_folder_info()

        while True:
            try:
                print("\n📋 입력 방식 선택:")
                print("1) YouTube URL 입력 (스트림 모드 - 빠른 시작)")
                print("2) YouTube URL 입력 (다운로드 모드 - ./downloads 폴더에 저장)")
                print("3) Downloads 폴더 파일 선택 (다운로드된 영상 + 수동으로 넣은 MP4)")
                print("4) 종료")
                print(f"\n💡 팁: MP4 파일을 ./downloads 폴더에 넣으면 3번에서 선택할 수 있습니다!")
                print(f"📁 영상 저장: ./mp4_output 폴더에 자동 저장됩니다!")

                choice = input("\n선택하세요 (1-4): ").strip()

                if choice == '1':
                    return self.setup_youtube_stream_mode()
                elif choice == '2':
                    return self.setup_youtube_download_mode()
                elif choice == '3':
                    return self.setup_downloads_folder_mode()
                elif choice == '4':
                    print("👋 프로그램을 종료합니다.")
                    return None
                else:
                    print("❌ 1-4 중에서 선택해주세요.")

            except KeyboardInterrupt:
                print(f"\n👋 프로그램을 종료합니다.")
                return None
            except Exception as e:
                print(f"❌ 입력 오류: {e}")

    def setup_youtube_stream_mode(self):
        """YouTube 스트림 모드 설정"""
        print("\n📡 YouTube 스트림 모드")
        print("✅ 장점: 빠른 시작, 네트워크 절약")
        print("⚠️ 단점: 시간 조절 제한")
        print("💾 저장: mp4_output 폴더에 자동 저장")

        youtube_url = self.get_youtube_url()
        if not youtube_url:
            return None

        return self.video_manager.init_from_youtube_url(youtube_url, mode='stream')

    def setup_youtube_download_mode(self):
        """YouTube 다운로드 모드 설정"""
        print("\n📥 YouTube 다운로드 모드")
        print("✅ 장점: 완전한 시간 조절, 화살표 키 지원")
        print("⚠️ 단점: 다운로드 시간 필요")
        print("💡 중단: 다운로드 중 Ctrl+C로 언제든지 중단 가능")
        print("💾 저장: downloads + mp4_output 양쪽 모두 저장")

        youtube_url = self.get_youtube_url()
        if not youtube_url:
            return None

        print("\n📊 품질 선택:")
        print("1) 720p (권장)")
        print("2) 1080p")
        print("3) 480p")
        print("4) 최고 품질")

        while True:
            try:
                quality_choice = input("품질 선택 (Enter=720p): ").strip() or '1'
                quality_map = {'1': '720p', '2': '1080p', '3': '480p', '4': 'best'}
                if quality_choice in quality_map:
                    quality = quality_map[quality_choice]
                    break
                else:
                    print("❌ 1-4 중에서 선택해주세요.")
            except KeyboardInterrupt:
                return None

        try:
            confirm = input("다운로드를 시작하시겠습니까? (y/n): ").lower()
            if confirm not in ['y', 'yes', '예', '']:
                return None
        except KeyboardInterrupt:
            return None

        return self.video_manager.init_from_youtube_url(youtube_url, mode='download', quality=quality)

    def setup_downloads_folder_mode(self):
        """downloads 폴더에서 파일 선택하여 재생"""
        download_dir = os.path.join(os.getcwd(), "downloads")

        if not os.path.exists(download_dir):
            print("❌ downloads 폴더가 없습니다.")
            print("💡 downloads 폴더를 생성하고 MP4 파일을 넣어주세요.")
            os.makedirs(download_dir, exist_ok=True)
            return None

        mp4_files = [f for f in os.listdir(download_dir) if f.endswith('.mp4')]

        if not mp4_files:
            print("❌ downloads 폴더에 MP4 파일이 없습니다.")
            print("💡 다음 방법으로 MP4 파일을 추가할 수 있습니다:")
            print("   1) 2번 메뉴에서 YouTube 영상 다운로드")
            print("   2) MP4 파일을 ./downloads 폴더에 직접 복사")
            return None

        print(f"\n📁 Downloads 폴더: {os.path.abspath(download_dir)}")
        print("📹 사용 가능한 영상 (다운로드 + 수동 추가):")
        print("💾 선택한 영상은 mp4_output 폴더에도 저장됩니다!")

        for i, file in enumerate(mp4_files, 1):
            file_path = os.path.join(download_dir, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))

            # 파일명으로 다운로드 파일인지 구분
            if "_" in file and file.count("_") >= 2:
                file_type = "📥"
            else:
                file_type = "📂"

            print(f"   {i}) {file_type} {file[:45]}{'...' if len(file) > 45 else ''}")
            print(f"      크기: {file_size:.1f}MB, 날짜: {mod_time.strftime('%Y-%m-%d %H:%M')}")

        try:
            while True:
                choice = input(f"\n파일 선택 (1-{len(mp4_files)}): ").strip()
                try:
                    index = int(choice) - 1
                    if 0 <= index < len(mp4_files):
                        selected_file = os.path.join(download_dir, mp4_files[index])
                        print(f"✅ 선택됨: {mp4_files[index]}")
                        return self.video_manager.init_from_file(selected_file)
                    else:
                        print(f"❌ 1-{len(mp4_files)} 범위에서 선택해주세요.")
                except ValueError:
                    print("❌ 숫자를 입력해주세요.")
        except KeyboardInterrupt:
            return None

    def get_youtube_url(self):
        """YouTube URL 입력받기"""
        default_url = self.config.get('youtube', {}).get('url', '')
        if default_url:
            print(f"기본 URL: {default_url}")

        while True:
            try:
                youtube_url = input("YouTube URL 입력 (Enter=기본값): ").strip()
                if not youtube_url and default_url:
                    youtube_url = default_url
                elif not youtube_url:
                    print("❌ URL을 입력해주세요.")
                    continue

                if 'youtube.com' in youtube_url or 'youtu.be' in youtube_url:
                    return youtube_url
                else:
                    print("❌ 올바른 YouTube URL을 입력해주세요.")
            except KeyboardInterrupt:
                return None

    def init_ai_model(self):
        try:
            weights_path = self.config['model']['weights_path']
            config_path = self.config['model']['config_path']
            classes_path = self.config['model']['classes_path']

            if not all(os.path.exists(p) for p in [weights_path, config_path, classes_path]):
                self.logger.warning("AI 모델 파일이 없습니다. 기본 처리 모드로 실행됩니다.")
                return False

            try:
                self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            except Exception:
                self.net = cv2.dnn.readNet(weights_path, config_path)

            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            with open(classes_path, 'r', encoding='utf-8') as f:
                self.classes = [line.strip() for line in f if line.strip()]

            layer_names = self.net.getLayerNames()
            unconnected = self.net.getUnconnectedOutLayers()
            self.output_layers = [layer_names[i - 1] for i in unconnected]

            self.logger.info(f"✅ AI 모델 로드 완료: {len(self.classes)}개 클래스")
            return True

        except Exception as e:
            self.logger.error(f"AI 모델 초기화 실패: {e}")
            return False

    def filter_vehicles_by_lane(self, vehicle_detections, frame_width, frame_height, frame):
        """차선 기반 차량 필터링"""
        lanes = self.lane_detector.detect_lanes(frame)
        if lanes is None:
            return []

        center_points = self.lane_detector.calculate_lane_center_points(lanes, frame_height)
        if not center_points:
            return []

        filtered = []
        lane_width = 120
        min_area = (frame_width * frame_height) * 0.008

        for det in vehicle_detections:
            if det['class_name'] not in ['car', 'truck', 'bus', 'motorbike', 'bicycle']:
                continue

            x, y, w, h = det['box']
            vehicle_center_x = x + w / 2
            vehicle_center_y = y + h / 2

            if vehicle_center_y < frame_height * 0.6:
                continue

            closest_point = min(center_points, key=lambda p: abs(p[1] - vehicle_center_y))
            lane_center_x = closest_point[0]

            if abs(vehicle_center_x - lane_center_x) < lane_width / 2 and (w * h) > min_area:
                det['in_zone'] = True
                det['front_vehicle'] = True
                filtered.append(det)

        return filtered

    def filter_traffic_light_detections(self, detections):
        """✅ 신호등 오탐지 필터링 - 세로로 긴 박스 제거"""
        filtered = []

        for det in detections:
            if det['class_name'] != 'traffic light':
                filtered.append(det)
                continue

            x, y, w, h = det['box']
            aspect_ratio = w / h if h > 0 else 0

            # 가로세로 비율이 0.3 이상인 신호등만 유효 (세로로 너무 긴 것 제외)
            # 일반적인 신호등은 가로가 세로보다 길거나 비슷함
            if aspect_ratio >= 0.3:
                filtered.append(det)
            else:
                self.logger.debug(f"세로형 신호등 필터링: 비율={aspect_ratio:.2f}")

        return filtered

    def detect_objects_optimized(self, frame):
        """최적화된 객체 감지"""
        start_time = time.time()

        height, width = frame.shape[:2]
        input_size = int(self.config['model'].get('input_size', 416))

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (input_size, input_size), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes, confidences, class_ids = [], [], []
        conf_th = float(self.config['detection']['confidence_threshold'])
        tl_conf_th = float(self.config['detection'].get('traffic_light_confidence_threshold', conf_th))
        nms_th = float(self.config['detection']['nms_threshold'])

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                if scores.size == 0:
                    continue

                class_id = int(np.argmax(scores))
                if class_id < 0 or class_id >= len(self.classes):
                    continue

                confidence = float(scores[class_id])
                class_name = self.classes[class_id]

                min_conf = tl_conf_th if class_name == 'traffic light' else conf_th

                if confidence > min_conf:
                    cx = int(detection[0] * width)
                    cy = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(cx - w / 2)
                    y = int(cy - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_th, nms_th)

        detections, vehicle_dets, tl_dets = [], [], []
        if len(indexes) > 0:
            for i in indexes.flatten():
                class_name = self.classes[class_ids[i]]
                det = {
                    'class_name': class_name,
                    'confidence': float(confidences[i]),
                    'box': boxes[i],
                    'timestamp': time.time()
                }

                detections.append(det)

                if class_name == 'traffic light':
                    tl_dets.append(det)
                    self.detection_stats['traffic_lights'] += 1

                if class_name in ['car', 'truck', 'bus', 'motorbike', 'bicycle']:
                    vehicle_dets.append(det)
                    self.detection_stats['vehicles'] += 1

        self.detection_stats['total_detections'] += len(indexes) if len(indexes) > 0 else 0

        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)

        # ✅ 신호등 오탐지 필터링 적용
        filtered_detections = self.filter_traffic_light_detections(detections)
        filtered_tl_dets = [d for d in filtered_detections if d['class_name'] == 'traffic light']

        filtered_vehicle_dets = self.filter_vehicles_by_lane(vehicle_dets, width, height, frame)
        tracked_vehicles = self.vehicle_tracker.update(filtered_vehicle_dets) if filtered_vehicle_dets else []

        if filtered_tl_dets and self.config['traffic_light']['enable_detection']:
            self.analyze_traffic_light_colors_fast(frame, filtered_tl_dets)

        return filtered_detections, tracked_vehicles

    def analyze_traffic_light_colors_fast(self, frame, tl_detections):
        for det in tl_detections:
            x, y, w, h = det['box']
            roi = frame[max(0, y):min(frame.shape[0], y + h),
                  max(0, x):min(frame.shape[1], x + w)]

            if roi.size > 100:
                detected_color = self.traffic_light_detector.detect_traffic_light_color(roi)
                change = self.traffic_light_detector.update_color_history(detected_color)

                if change:
                    self.handle_traffic_light_change(change)

    def handle_traffic_light_change(self, change_info):
        log_message = "신호가 변경되었습니다"
        self.logger.info(f"🚦 신호등 변화 감지")
        self._tts_announce_if_needed(log_message)
        self.detection_stats['traffic_light_changes'] += 1

    def log_vehicle_departure(self, track):
        if track.get('departure_detected', False):
            log_message = "앞의 차량이 출발하였습니다"
            self.logger.info(f"🚗 앞차 출발: ID{track['id']} ({track['class_name']})")
            self.detection_stats['vehicle_departures'] += 1
            self._tts_announce_if_needed(log_message)

    def _tts_announce_if_needed(self, msg):
        if not self.tts_system:
            return

        msg_hash = hashlib.md5(msg.encode()).hexdigest()
        if msg in self.TTS_ALLOWED and msg_hash not in self.tts_message_hashes:
            try:
                self.tts_system.play_situation_from_txt(msg, keyword=msg)
                self.tts_message_hashes.add(msg_hash)
                if len(self.tts_message_hashes) > 1000:
                    self.tts_message_hashes.pop()
            except Exception as e:
                self.logger.error(f"TTS 안내 실패: {e}")

    def draw_optimized_overlay(self, frame, detections, tracked_vehicles):
        """✅ 모드별 최적화된 오버레이 (불필요한 UI 제거)"""
        overlay = frame.copy()
        height, width = frame.shape[:2]
        mode_info = self.video_manager.get_mode_info()

        # 차선 오버레이
        lanes = self.lane_detector.last_valid_lanes
        center_points = self.lane_detector.calculate_lane_center_points(lanes, height) if lanes else None

        if lanes:
            if lanes.get('left'):
                x1, y1, x2, y2 = lanes['left']
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if lanes.get('right'):
                x1, y1, x2, y2 = lanes['right']
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if center_points:
            for i in range(len(center_points) - 1):
                cv2.line(overlay, center_points[i], center_points[i + 1], (0, 255, 255), 3)

        # 검출된 차량들 표시
        for detection in detections:
            x, y, w, h = detection['box']
            class_name = detection['class_name']
            confidence = detection['confidence']

            if class_name in ['car', 'truck', 'bus', 'motorbike', 'bicycle']:
                vehicle_center_y = y + h / 2
                if vehicle_center_y > height * 0.6:
                    colors = {
                        'car': (0, 255, 0),
                        'truck': (0, 255, 255),
                        'bus': (255, 255, 0),
                        'motorbike': (255, 0, 255),
                        'bicycle': (100, 255, 255)
                    }
                    color = colors.get(class_name, (0, 255, 0))
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                    conf_text = f"{class_name}: {confidence:.2f}"
                    cv2.putText(overlay, conf_text, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            elif class_name == 'traffic light':
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), 2)
                current_color = self.traffic_light_detector.last_stable_color
                if current_color:
                    signal_colors = {'red': (0, 0, 255), 'yellow': (0, 255, 255), 'green': (0, 255, 0)}
                    signal_color = signal_colors.get(current_color, (255, 255, 255))
                    cv2.circle(overlay, (x + w // 2, y - 15), 8, signal_color, -1)

        # 추적된 차량 표시
        for tr in tracked_vehicles:
            x, y, w, h = tr['bbox']
            track_id = tr['id']

            if tr.get('departure_detected', False):
                track_color = (0, 255, 255)
                status_text = "DEPARTED"
            elif tr.get('is_moving', False):
                track_color = (255, 255, 0)
                status_text = "MOVING"
            else:
                track_color = (0, 255, 0)
                status_text = "WAITING"

            cv2.rectangle(overlay, (x, y), (x + w, y + h), track_color, 3)
            id_text = f"ID-{track_id} ({status_text})"

            (text_w, text_h), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (x, y - text_h - 8), (x + text_w + 10, y), (0, 0, 0), -1)
            cv2.rectangle(overlay, (x, y - text_h - 8), (x + text_w + 10, y), track_color, 2)
            cv2.putText(overlay, id_text, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_color, 1)

        # ✅ 간소화된 상단 패널
        panel_h = 50 if mode_info['mode'] == 'stream' else 60
        panel_overlay = overlay.copy()
        cv2.rectangle(panel_overlay, (0, 0), (width, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, panel_overlay, 0.2, 0, overlay)
        cv2.line(overlay, (0, panel_h), (width, panel_h), (0, 255, 100), 2)

        # 모드별 정보 표시
        if mode_info['mode'] == 'stream':
            # ✅ 스트림 모드: 최소한의 정보만
            mode_text = "YouTube Live Stream → mp4_output"
            cv2.putText(overlay, mode_text, (15, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            current_time = self.video_manager.get_current_time()
            time_text = f"Live Time: {self.time_navigator.format_time(current_time)}"
            cv2.putText(overlay, time_text, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            # ✅ 파일 모드: 시간 조절 정보 포함
            mode_text = f"File Mode"
            cv2.putText(overlay, mode_text, (15, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            current_time = self.video_manager.get_current_time()
            total_time = self.video_manager.get_total_duration()
            progress = self.video_manager.get_progress()

            time_text = f"Time: {self.time_navigator.format_time(current_time)} / {self.time_navigator.format_time(total_time)} ({progress * 100:.1f}%)"
            cv2.putText(overlay, time_text, (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 시간 조절 UI (파일 모드에서만)
        if self.time_navigator.should_show_ui():
            self.draw_time_navigation_overlay(overlay, width, height)

        # 차량 감지 상태 (우측 상단)
        waiting_vehicles = len([t for t in tracked_vehicles if not t.get('is_moving', False)])
        moving_vehicles = len([t for t in tracked_vehicles if t.get('is_moving', False)])
        departed_vehicles = len([t for t in tracked_vehicles if t.get('departure_detected', False)])

        status_text = f"WAIT:{waiting_vehicles} MOVE:{moving_vehicles} DEPT:{departed_vehicles}"
        cv2.putText(overlay, status_text, (width - 300, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 신호등 상태
        if self.traffic_light_detector.last_stable_color:
            signal_text = f"SIGNAL: {self.traffic_light_detector.last_stable_color.upper()}"
            cv2.putText(overlay, signal_text, (width - 300, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # 일시정지 상태
        if self.paused:
            pause_text = "PAUSED"
            cv2.putText(overlay, pause_text, (width // 2 - 30, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # ✅ 모드별 컨트롤 힌트 (하단)
        if mode_info['can_seek']:
            control_text = "CONTROLS: [ARROWS] Navigate [SPACE] Pause [T] Jump Time [S] Screenshot [Q] Quit"
        else:
            control_text = "CONTROLS: [SPACE] Pause [S] Screenshot [Q] Quit"

        cv2.putText(overlay, control_text, (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return overlay

    def draw_time_navigation_overlay(self, overlay, width, height):
        """시간 탐색 UI 오버레이 (파일 모드 전용)"""
        nav_height = 100
        nav_y = height - nav_height - 50

        nav_overlay = overlay.copy()
        cv2.rectangle(nav_overlay, (50, nav_y), (width - 50, nav_y + nav_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, nav_overlay, 0.3, 0, overlay)
        cv2.rectangle(overlay, (50, nav_y), (width - 50, nav_y + nav_height), (0, 255, 255), 2)

        cv2.putText(overlay, "TIME NAVIGATION", (60, nav_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        jump_seconds = self.time_navigator.get_current_jump_seconds()
        jump_text = f"Jump Time: {jump_seconds}s"
        cv2.putText(overlay, jump_text, (60, nav_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        controls_text = "[ARROWS] Jump [A/D] 1s [T] Cycle"
        cv2.putText(overlay, controls_text, (60, nav_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def handle_keyboard_input(self, key):
        """✅ 모드별 최적화된 키보드 입력 처리"""
        if key == -1:
            return False

        mode_info = self.video_manager.get_mode_info()
        can_seek = mode_info['can_seek']

        # 공통 키
        if key == ord('q') or key == ord('Q'):
            self.logger.info("사용자 요청으로 종료합니다.")
            self.running = False
            return True

        elif key == ord('s') or key == ord('S'):
            screenshot_name = f"hybrid_{mode_info['mode']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            if self.current_frame is not None:
                cv2.imwrite(screenshot_name, self.current_frame)
                self.logger.info(f"📸 스크린샷 저장: {screenshot_name}")

        elif key == 32:  # SPACE
            self.paused = not self.paused
            status = "일시정지" if self.paused else "재개"
            self.logger.info(f"⏸️ {status}")

        # ✅ 시간 조절 키 (파일 모드에서만 처리)
        elif can_seek:
            if key == ord('t') or key == ord('T'):
                self.time_navigator.cycle_jump_time()

            elif key in [2424832, 65361]:  # LEFT ARROW
                jump_seconds = self.time_navigator.get_current_jump_seconds()
                if self.video_manager.jump_relative(-jump_seconds):
                    self.time_navigator.show_navigation_ui = True
                    self.time_navigator.ui_show_time = time.time()
                    self.logger.info(f"⬅️ {jump_seconds}초 뒤로")

            elif key in [2555904, 65363]:  # RIGHT ARROW
                jump_seconds = self.time_navigator.get_current_jump_seconds()
                if self.video_manager.jump_relative(jump_seconds):
                    self.time_navigator.show_navigation_ui = True
                    self.time_navigator.ui_show_time = time.time()
                    self.logger.info(f"➡️ {jump_seconds}초 앞으로")

            elif key in [2490368, 65362]:  # UP ARROW
                if self.video_manager.jump_relative(10):
                    self.time_navigator.show_navigation_ui = True
                    self.time_navigator.ui_show_time = time.time()
                    self.logger.info("⬆️ 10초 앞으로")

            elif key in [2621440, 65364]:  # DOWN ARROW
                if self.video_manager.jump_relative(-10):
                    self.time_navigator.show_navigation_ui = True
                    self.time_navigator.ui_show_time = time.time()
                    self.logger.info("⬇️ 10초 뒤로")

            elif key == ord('a') or key == ord('A'):
                if self.video_manager.jump_relative(-1):
                    self.time_navigator.show_navigation_ui = True
                    self.time_navigator.ui_show_time = time.time()
                    self.logger.info("⬅️ 1초 뒤로")

            elif key == ord('d') or key == ord('D'):
                if self.video_manager.jump_relative(1):
                    self.time_navigator.show_navigation_ui = True
                    self.time_navigator.ui_show_time = time.time()
                    self.logger.info("➡️ 1초 앞으로")

        # ✅ 스트림 모드에서는 시간 조절 키를 무시 (로그도 출력하지 않음)

        return False

    def run(self):
        """하이브리드 시스템 실행"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        if not self.get_user_choice():
            return False

        ai_ok = self.init_ai_model()
        if not ai_ok:
            self.logger.warning("AI 모델 없이 기본 처리 모드로 실행")

        self.running = True
        frame_count = 0

        if self.config['display'].get('show_preview', True):
            mode_info = self.video_manager.get_mode_info()
            window_name = f"Hybrid BlackBox - {mode_info['mode'].title()} Mode → mp4_output"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            window_w = self.config['display'].get('window_width', 1280)
            window_h = self.config['display'].get('window_height', 720)
            cv2.resizeWindow(window_name, window_w, window_h)

        video_fps = self.video_manager.get_fps()
        frame_delay = 1.0 / video_fps if video_fps > 0 else 1.0 / 30

        mode_info = self.video_manager.get_mode_info()
        self.logger.info(f"🎬 하이브리드 블랙박스 시작 ({mode_info['mode']} 모드)")
        self.logger.info(f"✅ 시간 조절: {'가능' if mode_info['can_seek'] else '불가능'}")
        self.logger.info(f"💾 영상 저장: mp4_output 폴더")

        try:
            while self.running:
                if not self.paused:
                    ret, frame = self.video_manager.read_frame()
                    if not ret or frame is None:
                        if mode_info['mode'] == 'stream':
                            self.logger.warning("⚠️ 스트림 연결이 끊어졌습니다.")
                            time.sleep(5)
                            continue
                        else:
                            self.logger.info("📹 비디오 처리 완료")
                            break

                    self.current_frame = frame.copy()

                    detections, tracked_vehicles = [], []
                    detection_interval = int(self.config['detection']['detection_interval'])

                    if ai_ok and self.net is not None:
                        if frame_count % detection_interval == 0:
                            detections, tracked_vehicles = self.detect_objects_optimized(frame)

                    # ✅ 최적화된 오버레이 적용
                    overlay_frame = self.draw_optimized_overlay(frame.copy(), detections, tracked_vehicles)

                    # ✅ mp4_output에 프레임 저장
                    self.video_manager.write_frame(overlay_frame)

                    # 차량 출발 감지 처리
                    for track in tracked_vehicles:
                        if track.get('departure_detected', False) and not track.get('logged', False):
                            self.log_vehicle_departure(track)
                            track['logged'] = True

                    frame_count += 1

                else:
                    # 일시정지 상태에서도 현재 프레임 표시
                    if self.current_frame is not None:
                        overlay_frame = self.draw_optimized_overlay(self.current_frame.copy(), [], [])
                    else:
                        time.sleep(0.1)
                        continue

                # 화면 표시
                if self.config['display'].get('show_preview', True):
                    cv2.imshow(window_name, overlay_frame)

                # 키보드 입력 처리
                key = cv2.waitKeyEx(1)
                if self.handle_keyboard_input(key):
                    break

                # 프레임 딜레이 (일시정지가 아닐 때만)
                if not self.paused:
                    time.sleep(frame_delay)

        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt 감지")
        except Exception as e:
            self.logger.error(f"실행 중 오류: {e}")
        finally:
            self.cleanup_resources()

        return True

    def signal_handler(self, signum, frame):
        """시그널 핸들러"""
        self.logger.info(f"시그널 {signum} 수신, 정상 종료합니다")
        self.running = False
        self.cleanup_resources()
        sys.exit(0)

    def cleanup_resources(self):
        """리소스 정리"""
        try:
            self.running = False

            # ✅ VideoWriter 해제 (mp4_output 저장 완료)
            if self.video_manager:
                self.video_manager.release()

            cv2.destroyAllWindows()

            if hasattr(self, 'tts_system') and self.tts_system:
                self.tts_system.shutdown()

        except Exception as e:
            self.logger.error(f"리소스 정리 실패: {e}")


def main():
    parser = argparse.ArgumentParser(description='하이브리드 블랙박스 시스템 v4.4 - 신호등 오탐지 해결 + mp4_output 저장')
    parser.add_argument('--config', default='hybrid_blackbox_config.json', help='설정 파일 경로')
    parser.add_argument('--mode', choices=['stream', 'download', 'file'], help='실행 모드')
    parser.add_argument('--url', help='YouTube URL')
    parser.add_argument('--file', help='MP4 파일 경로')
    parser.add_argument('--quality', default='720p', choices=['480p', '720p', '1080p', 'best'], help='다운로드 품질')
    parser.add_argument('--debug', action='store_true', help='디버그 모드')

    args = parser.parse_args()

    if not YT_DLP_AVAILABLE:
        print("❌ yt-dlp 라이브러리가 필요합니다.")
        print("설치 명령어: pip install yt-dlp")
        return 1

    try:
        blackbox = HybridBlackBoxSystem(args.config)

        # 명령행 인수로 모드 지정 시
        if args.mode:
            if args.mode in ['stream', 'download']:
                if not args.url:
                    print("❌ YouTube 모드에는 --url 인수가 필요합니다.")
                    return 1
                success = blackbox.video_manager.init_from_youtube_url(args.url, mode=args.mode, quality=args.quality)
            elif args.mode == 'file':
                if not args.file:
                    print("❌ 파일 모드에는 --file 인수가 필요합니다.")
                    return 1
                success = blackbox.video_manager.init_from_file(args.file)

            if not success:
                print("❌ 초기화 실패")
                return 1

        success = blackbox.run()
        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n👋 프로그램이 사용자에 의해 중단되었습니다.")
        return 0
    except Exception as e:
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print(f"❌ 오류 발생: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
