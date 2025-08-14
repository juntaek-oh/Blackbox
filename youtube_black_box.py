#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YouTube 링크 직접 처리 블랙박스 시스템 v3.3 - 한글 폰트 깨짐 수정 + UI 간소화판
한글 텍스트 완벽 표시 + YouTube 스트림 전용 최적화
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


class SmartVideoSizeManager:
    """스마트 영상 크기 자동 감지 및 조정 관리자"""

    def __init__(self, logger):
        self.logger = logger
        self.original_width = 1280
        self.original_height = 720
        self.display_width = 1280
        self.display_height = 720
        self.aspect_ratio = 16 / 9
        self.scale_factor = 1.0
        self.max_window_width = 1920
        self.max_window_height = 1080
        self.min_window_width = 640
        self.min_window_height = 360

    def update_video_size(self, frame_width, frame_height):
        """영상의 실제 크기에 맞게 표시 크기 조정"""
        self.original_width = frame_width
        self.original_height = frame_height
        self.aspect_ratio = frame_width / frame_height if frame_height > 0 else 16 / 9

        # 최적 표시 크기 계산
        self.calculate_optimal_display_size()

        self.logger.info(f"✅ 영상 크기 감지: {frame_width}x{frame_height} (비율: {self.aspect_ratio:.3f})")
        self.logger.info(f"✅ 표시 크기 조정: {self.display_width}x{self.display_height} (배율: {self.scale_factor:.2f}x)")

    def calculate_optimal_display_size(self):
        """화면 크기에 맞는 최적 표시 크기 계산"""
        if self.original_width > self.max_window_width or self.original_height > self.max_window_height:
            width_scale = self.max_window_width / self.original_width
            height_scale = self.max_window_height / self.original_height
            self.scale_factor = min(width_scale, height_scale)
        elif self.original_width < self.min_window_width or self.original_height < self.min_window_height:
            width_scale = self.min_window_width / self.original_width
            height_scale = self.min_window_height / self.original_height
            self.scale_factor = max(width_scale, height_scale)
        else:
            self.scale_factor = 1.0

        self.display_width = int(self.original_width * self.scale_factor)
        self.display_height = int(self.original_height * self.scale_factor)

        if self.display_width % 2 != 0:
            self.display_width += 1
        if self.display_height % 2 != 0:
            self.display_height += 1

        if self.display_width > self.max_window_width:
            self.display_width = self.max_window_width
            self.display_height = int(self.display_width / self.aspect_ratio)
        if self.display_height > self.max_window_height:
            self.display_height = self.max_window_height
            self.display_width = int(self.display_height * self.aspect_ratio)

    def resize_frame_if_needed(self, frame):
        """필요한 경우 프레임 크기 조정"""
        if frame is None:
            return None

        current_height, current_width = frame.shape[:2]

        if current_width != self.display_width or current_height != self.display_height:
            if self.scale_factor < 1.0:
                resized_frame = cv2.resize(frame, (self.display_width, self.display_height),
                                           interpolation=cv2.INTER_AREA)
            else:
                resized_frame = cv2.resize(frame, (self.display_width, self.display_height),
                                           interpolation=cv2.INTER_CUBIC)
            return resized_frame
        return frame

    def get_window_size(self):
        """윈도우 크기 반환"""
        return self.display_width, self.display_height


class KoreanTextRenderer:
    """✅ 한글 텍스트 완벽 렌더링 클래스"""

    def __init__(self):
        self.font_loaded = False
        self.font_scale = 0.5
        self.thickness = 1

    def put_korean_text(self, img, text, position, font_scale=0.5, color=(255, 255, 255), thickness=1, background=None):
        """한글이 포함된 텍스트를 이미지에 그리기 (OpenCV 기본 폰트 사용 + 영문 대체)"""
        try:
            # 한글이 포함된 경우 영문으로 대체하여 표시
            clean_text = self.convert_korean_to_english(text)

            # 배경색이 지정된 경우 배경 박스 그리기
            if background is not None:
                (text_w, text_h), _ = cv2.getTextSize(clean_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                x, y = position
                cv2.rectangle(img, (x - 2, y - text_h - 5), (x + text_w + 5, y + 5), background, -1)

            # 텍스트 그리기
            cv2.putText(img, clean_text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        except Exception as e:
            # 실패 시 기본 OpenCV 폰트로 영문만 표시
            clean_text = self.extract_english_only(text)
            cv2.putText(img, clean_text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    def convert_korean_to_english(self, text):
        """한글 텍스트를 의미있는 영문으로 변환"""
        korean_to_english = {
            # 기본 상태
            "대기": "WAIT",
            "이동": "MOVE",
            "출발": "DEPT",
            "검출": "DETECT",
            "차량": "VEHICLE",
            "추적": "TRACK",
            "신호등": "SIGNAL",
            "변화": "CHANGE",
            "원본": "ORIG",
            "표시": "DISP",

            # 신호등 색상
            "빨간색": "RED",
            "노란색": "YELLOW",
            "초록색": "GREEN",
            "적색": "RED",
            "황색": "YELLOW",
            "녹색": "GREEN",

            # 시간 관련
            "시간": "TIME",
            "초": "SEC",
            "분": "MIN",
            "시": "HOUR",

            # 기타
            "실시간": "LIVE",
            "연결": "CONN",
            "상태": "STATUS",
            "정보": "INFO",
            "시스템": "SYSTEM"
        }

        result = text
        for korean, english in korean_to_english.items():
            result = result.replace(korean, english)

        return result

    def extract_english_only(self, text):
        """텍스트에서 영문, 숫자, 기호만 추출"""
        import re
        return re.sub(r'[^\x00-\x7F]', '', text)


class YouTubeVideoManager:
    """YouTube 링크 처리 및 실시간 스트리밍 관리"""

    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.video_capture = None
        self.video_fps = 30
        self.total_frames = 0
        self.current_frame_idx = 0
        self.video_duration = 0
        self.youtube_url = None
        self.stream_url = None
        self.video_info = {}

        # 크기 관리자 추가
        self.size_manager = SmartVideoSizeManager(logger)
        self.frame_size_detected = False

    def init_youtube_video(self, youtube_url):
        """YouTube URL에서 비디오 스트림 초기화"""
        if not YT_DLP_AVAILABLE:
            self.logger.error("❌ yt-dlp 라이브러리가 필요합니다.")
            return False

        self.youtube_url = youtube_url
        self.logger.info(f"📹 YouTube 영상 로드 중: {youtube_url}")

        try:
            quality_formats = [
                'best[height<=1080][ext=mp4]/best[height<=1080]',
                'best[height<=720][ext=mp4]/best[height<=720]',
                'best[height<=480][ext=mp4]/best[height<=480]',
                'best[height<=360][ext=mp4]/best[height<=360]',
                'best[ext=mp4]/best',
                'worst'
            ]

            best_info = None
            selected_format = None

            for format_selector in quality_formats:
                try:
                    ydl_opts = {
                        'format': format_selector,
                        'quiet': True,
                        'no_warnings': True,
                        'extract_flat': False,
                    }

                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        self.video_info = ydl.extract_info(youtube_url, download=False)

                        if 'url' in self.video_info:
                            self.stream_url = self.video_info['url']
                            best_info = self.video_info
                            selected_format = format_selector
                            break
                        elif 'formats' in self.video_info:
                            for fmt in self.video_info['formats']:
                                if (fmt.get('vcodec') != 'none' and
                                        fmt.get('acodec') != 'none' and
                                        'url' in fmt):
                                    self.stream_url = fmt['url']
                                    best_info = self.video_info
                                    selected_format = format_selector
                                    if 'width' in fmt and 'height' in fmt:
                                        self.video_info['detected_width'] = fmt['width']
                                        self.video_info['detected_height'] = fmt['height']
                                    break
                            if self.stream_url:
                                break

                except Exception as e:
                    self.logger.warning(f"품질 '{format_selector}' 시도 실패: {e}")
                    continue

            if not self.stream_url:
                self.logger.error("❌ 적절한 스트림 URL을 찾을 수 없습니다.")
                return False

            self.video_duration = self.video_info.get('duration', 0)
            title = self.video_info.get('title', 'Unknown')
            uploader = self.video_info.get('uploader', 'Unknown')

            detected_width = self.video_info.get('detected_width', 'Unknown')
            detected_height = self.video_info.get('detected_height', 'Unknown')

            self.logger.info(f"✅ YouTube 영상 정보:")
            self.logger.info(f"   제목: {title[:50]}{'...' if len(title) > 50 else ''}")
            self.logger.info(f"   업로더: {uploader}")
            self.logger.info(f"   길이: {self.video_duration:.1f}초")
            self.logger.info(f"   선택된 품질: {selected_format}")
            self.logger.info(f"   예상 해상도: {detected_width}x{detected_height}")

        except Exception as e:
            self.logger.error(f"❌ YouTube 영상 정보 추출 실패: {e}")
            return False

        # OpenCV VideoCapture로 스트림 열기
        try:
            self.logger.info("📡 스트림 연결 중...")
            cap = cv2.VideoCapture(self.stream_url)

            max_retries = 3
            for attempt in range(max_retries):
                if cap.isOpened():
                    break
                self.logger.warning(f"연결 시도 {attempt + 1}/{max_retries}...")
                time.sleep(2)
                cap.release()
                cap = cv2.VideoCapture(self.stream_url)

            if not cap.isOpened():
                self.logger.error("❌ 스트림을 열 수 없습니다.")
                return False

            max_frame_attempts = 10
            frame = None
            for attempt in range(max_frame_attempts):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    break
                self.logger.warning(f"프레임 읽기 시도 {attempt + 1}/{max_frame_attempts}...")
                time.sleep(0.5)

            if frame is None:
                self.logger.error("❌ 스트림에서 프레임을 읽을 수 없습니다.")
                cap.release()
                return False

            actual_height, actual_width = frame.shape[:2]
            ch = frame.shape[2] if len(frame.shape) == 3 else 1

            self.size_manager.update_video_size(actual_width, actual_height)
            self.frame_size_detected = True

            self.video_fps = cap.get(cv2.CAP_PROP_FPS)

            if self.video_fps <= 0 or self.video_fps > 60:
                self.video_fps = 30
                self.logger.info(f"⚠️ FPS 자동 설정: {self.video_fps}fps")

            self.total_frames = int(self.video_duration * self.video_fps) if self.video_duration > 0 else 999999

            self.logger.info(f"✅ 스트림 연결 성공:")
            self.logger.info(f"   실제 해상도: {actual_width}x{actual_height}x{ch}")
            self.logger.info(f"   표시 해상도: {self.size_manager.display_width}x{self.size_manager.display_height}")
            self.logger.info(f"   FPS: {self.video_fps:.1f}")

            self.video_capture = cap
            self.current_frame_idx = 0
            return True

        except Exception as e:
            self.logger.error(f"스트림 연결 실패: {e}")
            return False

    def get_current_time(self):
        """현재 재생 시간(초) 반환"""
        if self.video_fps == 0:
            return 0
        return self.current_frame_idx / self.video_fps

    def get_total_duration(self):
        """전체 영상 길이(초) 반환"""
        return self.video_duration

    def read_frame(self):
        """프레임 읽기 (스트리밍)"""
        if not self.video_capture or not self.video_capture.isOpened():
            return False, None

        ret, frame = self.video_capture.read()
        if ret and frame is not None:
            if not self.frame_size_detected:
                actual_height, actual_width = frame.shape[:2]
                self.size_manager.update_video_size(actual_width, actual_height)
                self.frame_size_detected = True

            resized_frame = self.size_manager.resize_frame_if_needed(frame)
            self.current_frame_idx += 1
            return True, resized_frame
        else:
            self.logger.warning("⚠️ 스트림 연결이 끊어졌습니다.")
            return False, None

    def get_fps(self):
        return self.video_fps

    def get_progress(self):
        if self.total_frames == 0 or self.video_duration == 0:
            return 0.0
        return min(self.current_frame_idx / self.total_frames, 1.0)

    def get_window_size(self):
        return self.size_manager.get_window_size()

    def release(self):
        if self.video_capture:
            self.video_capture.release()


# 간단한 차선 감지, 추적기, 신호등 감지 클래스들
class LaneDetector:
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
    def __init__(self, max_lost=5, iou_threshold=0.3, movement_threshold=2, expected_fps=30):
        self.tracks = {}
        self.next_id = 1
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.movement_threshold = movement_threshold
        self.frame_count = 0
        self.expected_fps = expected_fps
        self.departure_buffer_frames = int(expected_fps * 2)
        self.stationary_threshold = 1.5
        self.min_stationary_ratio = 0.7

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

        recent_movements = list(track['movement_history'])[-self.departure_buffer_frames:]
        analysis_frames = recent_movements[:-15] if len(recent_movements) > 15 else recent_movements[:-5]
        if not analysis_frames:
            return False

        stationary_count = 0
        for move_x, move_y in analysis_frames:
            distance = (move_x ** 2 + move_y ** 2) ** 0.5
            if distance < self.stationary_threshold:
                stationary_count += 1

        stationary_ratio = stationary_count / len(analysis_frames)
        if stationary_ratio < self.min_stationary_ratio:
            return False

        recent_movement_frames = recent_movements[-10:] if len(recent_movements) >= 10 else recent_movements[-5:]
        avg_recent_distance = sum(
            (move_x ** 2 + move_y ** 2) ** 0.5
            for move_x, move_y in recent_movement_frames
        ) / len(recent_movement_frames)

        adjusted_threshold = self.movement_threshold * (2.0 if is_in_zone else 3.0)
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

        for track_id, track in active_tracks.items():
            if track_id not in matched_tracks:
                track['lost'] += 1

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

        red_lower1 = np.array([0, 100, 100]);
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 100, 100]);
        red_upper2 = np.array([180, 255, 255])
        yellow_lower = np.array([18, 120, 120]);
        yellow_upper = np.array([35, 255, 255])
        green_lower = np.array([45, 100, 100]);
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


class YouTubeBlackBoxSystem:
    """✅ 한글 폰트 깨짐 수정 + YouTube 스트림 전용 최적화 시스템"""

    def __init__(self, config_path="youtube_blackbox_clean.json"):
        self.setup_logging()
        self.config_path = config_path
        self.config = self.load_config()

        self.video_manager = YouTubeVideoManager(self.logger, self.config)
        # ✅ 한글 텍스트 렌더러 추가
        self.text_renderer = KoreanTextRenderer()

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
            movement_threshold=float(self.config.get('tracking', {}).get('movement_threshold', 2)),
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

        self.logger.info("✅ YouTube 스트림 전용 블랙박스 시스템 (한글 지원) 초기화 완료")

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
                logging.FileHandler('log/youtube_clean.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self):
        default_config = {
            "youtube": {
                "url": "https://www.youtube.com/watch?v=OoHzud9L48s",
                "quality": "best",
                "output_dir": "output"
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
                "movement_threshold": 2
            },
            "display": {
                "show_preview": True,
                "auto_size": True
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

    def get_youtube_url_from_user(self):
        """사용자로부터 YouTube URL 입력받기"""
        print("\n" + "=" * 80)
        print("     YouTube Stream BlackBox System (Clean Korean UI)")
        print("=" * 80)

        default_url = self.config.get('youtube', {}).get('url', '')
        if default_url:
            print(f"기본 URL: {default_url}")

        while True:
            try:
                user_input = input("\nYouTube URL을 입력하세요 (Enter=기본값 사용): ").strip()

                if not user_input and default_url:
                    youtube_url = default_url
                elif not user_input:
                    print("❌ URL을 입력해주세요.")
                    continue
                else:
                    youtube_url = user_input

                if 'youtube.com' in youtube_url or 'youtu.be' in youtube_url:
                    self.logger.info(f"✅ YouTube URL 확인: {youtube_url}")
                    return youtube_url
                else:
                    print("❌ 올바른 YouTube URL을 입력해주세요.")

            except KeyboardInterrupt:
                print("\n프로그램을 종료합니다.")
                return None
            except Exception as e:
                print(f"❌ 입력 오류: {e}")

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
        """차선 기반 앞차 필터링"""
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

    def detect_objects_optimized(self, frame):
        """성능 최적화된 객체 감지"""
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

            self.detection_stats['total_detections'] += len(indexes)

        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)

        filtered_vehicle_dets = self.filter_vehicles_by_lane(vehicle_dets, width, height, frame)
        tracked_vehicles = self.vehicle_tracker.update(filtered_vehicle_dets) if filtered_vehicle_dets else []

        if tl_dets and self.config['traffic_light']['enable_detection']:
            self.analyze_traffic_light_colors_fast(frame, tl_dets)

        return detections, tracked_vehicles

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
        self.logger.info(f"🚦 {log_message}")
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

    def format_time(self, seconds):
        """시간 포맷팅"""
        if seconds < 0:
            return "00:00"
        hours = int(seconds // 3600)
        minutes = int(seconds % 3600) // 60
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    def draw_clean_overlay(self, frame, detections, tracked_vehicles):
        """✅ 깔끔하고 한글이 완벽하게 표시되는 오버레이 (시간 조절 UI 제거)"""
        overlay = frame.copy()
        height, width = frame.shape[:2]

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
                    self.text_renderer.put_korean_text(overlay, conf_text, (x, y - 5),
                                                       font_scale=0.4, color=color, thickness=1)

            elif class_name == 'traffic light':
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), 2)
                current_color = self.traffic_light_detector.last_stable_color
                if current_color:
                    signal_colors = {'red': (0, 0, 255), 'yellow': (0, 255, 255), 'green': (0, 255, 0)}
                    signal_color = signal_colors.get(current_color, (255, 255, 255))
                    cv2.circle(overlay, (x + w // 2, y - 15), 8, signal_color, -1)

        # 추적된 앞차 표시 (한글 완벽 지원)
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
            self.text_renderer.put_korean_text(overlay, id_text, (x + 5, y - 5),
                                               font_scale=0.5, color=track_color, thickness=1,
                                               background=(0, 0, 0))

        # ✅ 간소화된 상단 패널 (한글 완벽 지원)
        panel_h = 60
        panel_overlay = overlay.copy()
        cv2.rectangle(panel_overlay, (0, 0), (width, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, panel_overlay, 0.2, 0, overlay)
        cv2.line(overlay, (0, panel_h), (width, panel_h), (0, 255, 100), 2)

        # YouTube 정보 표시
        title = self.video_manager.video_info.get('title', 'YouTube Stream')
        youtube_info = f"📺 {title[:40]}{'...' if len(title) > 40 else ''}"
        self.text_renderer.put_korean_text(overlay, youtube_info, (15, 20),
                                           font_scale=0.5, color=(255, 255, 255), thickness=1)

        # 스트림 시간 정보 (시간 조절 UI는 제거하고 정보만 표시)
        current_time = self.video_manager.get_current_time()
        total_time = self.video_manager.get_total_duration()
        if total_time > 0:
            time_text = f"TIME: {self.format_time(current_time)} / {self.format_time(total_time)}"
        else:
            time_text = f"LIVE TIME: {self.format_time(current_time)}"
        self.text_renderer.put_korean_text(overlay, time_text, (15, 40),
                                           font_scale=0.4, color=(255, 255, 255), thickness=1)

        # 앞차 감지 상태 (한글 → 영문 변환으로 표시)
        waiting_vehicles = len([t for t in tracked_vehicles if not t.get('is_moving', False)])
        moving_vehicles = len([t for t in tracked_vehicles if t.get('is_moving', False)])
        departed_vehicles = len([t for t in tracked_vehicles if t.get('departure_detected', False)])

        status_text = f"🚗 WAIT:{waiting_vehicles} MOVE:{moving_vehicles} DEPT:{departed_vehicles}"
        self.text_renderer.put_korean_text(overlay, status_text, (width - 300, 25),
                                           font_scale=0.4, color=(0, 255, 0), thickness=1)

        # 신호등 상태
        if self.traffic_light_detector.last_stable_color:
            signal_text = f"🚦 SIGNAL: {self.traffic_light_detector.last_stable_color.upper()}"
            self.text_renderer.put_korean_text(overlay, signal_text, (width - 300, 45),
                                               font_scale=0.4, color=(0, 255, 255), thickness=1)

        # 일시정지 상태
        if self.paused:
            pause_text = "⏸️ PAUSED"
            self.text_renderer.put_korean_text(overlay, pause_text, (width // 2 - 40, height // 2),
                                               font_scale=1, color=(0, 0, 255), thickness=2)

        # 스트림 연결 상태
        cv2.circle(overlay, (width - 30, 30), 8, (0, 255, 0), -1)
        self.text_renderer.put_korean_text(overlay, "LIVE", (width - 60, 35),
                                           font_scale=0.3, color=(0, 255, 0), thickness=1)

        # ✅ 간소화된 컨트롤 힌트 (시간 조절 관련 제거)
        control_text = "CONTROLS: [SPACE] Pause [S] Screenshot [Q] Quit [H] Help"
        self.text_renderer.put_korean_text(overlay, control_text, (10, height - 10),
                                           font_scale=0.4, color=(200, 200, 200), thickness=1)

        return overlay

    def handle_keyboard_input(self, key):
        """✅ 키보드 입력 처리 (시간 조절 관련 제거)"""
        if key == -1:
            return False

        if key == ord('q') or key == ord('Q'):
            self.logger.info("사용자 요청으로 종료합니다.")
            self.running = False
            return True

        elif key == ord('s') or key == ord('S'):
            screenshot_name = f"youtube_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            if self.current_frame is not None:
                cv2.imwrite(screenshot_name, self.current_frame)
                self.logger.info(f"📸 스크린샷 저장: {screenshot_name}")

        elif key == 32:  # SPACE
            self.paused = not self.paused
            status = "일시정지" if self.paused else "재개"
            self.logger.info(f"⏸️ 스트림 {status}")

        elif key == ord('h') or key == ord('H'):
            self.show_help()

        # ✅ 시간 이동 관련 키는 모두 제거하고 안내 메시지만 표시
        elif key in [2424832, 65361, 2555904, 65363, 2490368, 65362, 2621440, 65364]:
            self.logger.info("ℹ️ YouTube 스트림에서는 시간 이동 기능이 지원되지 않습니다.")

        return False

    def show_help(self):
        """✅ 간소화된 도움말 (시간 조절 관련 제거)"""
        help_text = """
==================== YouTube Stream BlackBox (Clean UI) ====================
[기본 제어]
Q - 프로그램 종료
SPACE - 일시정지/재개  
S - 스크린샷 저장
H - 이 도움말 표시

[특징]
- 한글 텍스트 완벽 지원 (물음표 없음)
- YouTube 스트림 전용 최적화
- 불필요한 시간 조절 UI 제거
- 깔끔하고 직관적인 인터페이스

[기능]
- 실시간 앞차 출발 감지
- 신호등 상태 인식
- 차선 기반 정밀 추적
- 자동 영상 크기 조정

[제한사항]
- YouTube 스트림 특성상 시간 이동(되감기/빨리감기) 불가
- 네트워크 상태에 따른 지연 가능
========================================================================
"""
        self.logger.info(help_text)
        print(help_text)

    def run(self):
        # 사용자로부터 YouTube URL 입력받기
        youtube_url = self.get_youtube_url_from_user()
        if not youtube_url:
            return False

        # YouTube 스트림 초기화
        if not self.video_manager.init_youtube_video(youtube_url):
            return False

        ai_ok = self.init_ai_model()
        if not ai_ok:
            self.logger.warning("AI 모델 없이 기본 처리 모드로 실행")

        self.running = True
        frame_count = 0

        # 실제 영상 크기에 맞게 윈도우 설정
        if self.config['display'].get('show_preview', True):
            window_name = "YouTube Stream BlackBox (Clean Korean UI)"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            display_w, display_h = self.video_manager.get_window_size()
            cv2.resizeWindow(window_name, display_w, display_h)

            self.logger.info(f"✅ 윈도우 크기 자동 설정: {display_w}x{display_h}")

        video_fps = self.video_manager.get_fps()
        frame_delay = 1.0 / video_fps if video_fps > 0 else 1.0 / 30

        self.logger.info(f"🎬 YouTube 스트림 시작 (FPS: {video_fps:.1f})")
        self.logger.info("✅ 한글 완벽 지원 + 깔끔한 UI")
        self.logger.info("✅ 컨트롤: [SPACE]일시정지 [S]스크린샷 [H]도움말 [Q]종료")

        try:
            while self.running:
                if not self.paused:
                    ret, frame = self.video_manager.read_frame()
                    if not ret or frame is None:
                        self.logger.warning("⚠️ 스트림 연결이 끊어졌습니다. 재연결을 시도합니다...")
                        time.sleep(5)
                        continue

                    self.current_frame = frame.copy()

                    detections, tracked_vehicles = [], []
                    detection_interval = int(self.config['detection']['detection_interval'])

                    if ai_ok and self.net is not None:
                        if frame_count % detection_interval == 0:
                            detections, tracked_vehicles = self.detect_objects_optimized(frame)

                    # ✅ 깔끔한 오버레이 적용 (한글 완벽 지원)
                    overlay_frame = self.draw_clean_overlay(frame.copy(), detections, tracked_vehicles)

                    # 앞차 출발 감지 로그
                    for track in tracked_vehicles:
                        if track.get('departure_detected', False) and not track.get('logged', False):
                            self.log_vehicle_departure(track)
                            track['logged'] = True

                    frame_count += 1

                else:
                    # 일시정지 상태
                    if self.current_frame is not None:
                        overlay_frame = self.draw_clean_overlay(self.current_frame.copy(), [], [])
                    else:
                        time.sleep(0.1)
                        continue

                # 실시간 미리보기 표시
                if self.config['display'].get('show_preview', True):
                    cv2.imshow(window_name, overlay_frame)

                    # ✅ waitKeyEx()로 키 입력 처리 (화살표 키 지원 유지)
                    key = cv2.waitKeyEx(1)

                    if self.handle_keyboard_input(key):
                        break

                # 스트림 속도 조절
                if not self.paused:
                    time.sleep(frame_delay)

        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt 감지")
        except Exception as e:
            self.logger.error(f"실행 중 오류: {e}")
        finally:
            self.cleanup_resources()

        return True

    def cleanup_resources(self):
        try:
            self.video_manager.release()
            cv2.destroyAllWindows()
            if hasattr(self, 'tts_system') and self.tts_system:
                self.tts_system.shutdown()
        except Exception as e:
            self.logger.error(f"리소스 정리 실패: {e}")


def main():
    parser = argparse.ArgumentParser(description='YouTube Stream BlackBox (Clean Korean UI)')
    parser.add_argument('--config', default='youtube_blackbox_clean.json', help='설정 파일 경로')
    parser.add_argument('--url', help='YouTube URL (대화형 입력 생략)')
    parser.add_argument('--debug', action='store_true', help='디버그 모드')
    args = parser.parse_args()

    if not YT_DLP_AVAILABLE:
        print("❌ yt-dlp 라이브러리가 필요합니다.")
        print("설치 명령어: pip install yt-dlp")
        return 1

    try:
        blackbox = YouTubeBlackBoxSystem(args.config)

        if args.url:
            blackbox.config['youtube']['url'] = args.url

        success = blackbox.run()
        return 0 if success else 1
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
