#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MP4 영상 전용 블랙박스 시스템 v2.3 - 화살표 키 완전 수정판
cv2.waitKeyEx()를 사용하여 모든 플랫폼에서 특수 키 입력 지원
앞차 출발 오탐 방지 완전판 + 시간 탐색 기능
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

from tts_config import LOGGING_LEVEL, LOGGING_FORMAT
from tts_settings import TTSNavigationSystem

# 전역 로깅 설정
logging.basicConfig(level=getattr(logging, LOGGING_LEVEL), format=LOGGING_FORMAT)


class TimeNavigator:
    """시간 탐색 기능 클래스"""

    def __init__(self, logger):
        self.logger = logger
        self.jump_seconds = [5, 10, 30, 60, 120, 300]  # 점프 가능한 초 단위
        self.current_jump_index = 2  # 기본 30초
        self.show_navigation_ui = False
        self.ui_show_time = 0
        self.ui_display_duration = 3.0  # UI 표시 시간 (초)

    def get_current_jump_seconds(self):
        """현재 점프 설정값 반환"""
        return self.jump_seconds[self.current_jump_index]

    def cycle_jump_time(self):
        """점프 시간 순환 (5초 -> 10초 -> 30초 -> 60초 -> 120초 -> 300초 -> 5초...)"""
        self.current_jump_index = (self.current_jump_index + 1) % len(self.jump_seconds)
        self.show_navigation_ui = True
        self.ui_show_time = time.time()
        jump_time = self.get_current_jump_seconds()
        self.logger.info(f"⏱️ 점프 시간 설정: {jump_time}초")
        return jump_time

    def should_show_ui(self):
        """UI 표시 여부 확인"""
        if not self.show_navigation_ui:
            return False
        if time.time() - self.ui_show_time > self.ui_display_duration:
            self.show_navigation_ui = False
            return False
        return True

    def format_time(self, seconds):
        """시간을 MM:SS 형식으로 포맷팅"""
        if seconds < 0:
            return "00:00"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


class VideoManager:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.video_capture = None
        self.video_fps = 30
        self.total_frames = 0
        self.current_frame_idx = 0
        self.video_duration = 0  # 전체 영상 길이 (초)

    def init_video(self, video_path):
        """MP4 파일 초기화"""
        if not os.path.exists(video_path):
            self.logger.error(f"❌ 비디오 파일을 찾을 수 없습니다: {video_path}")
            return False

        self.logger.info(f"📹 MP4 파일 로드 중: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            self.logger.error("❌ 비디오 파일을 열 수 없습니다")
            return False

        # 첫 프레임 확인
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            self.logger.error("❌ 비디오 파일에서 프레임을 읽을 수 없습니다")
            cap.release()
            return False

        # 비디오 정보 가져오기
        h, w = frame.shape[:2]
        ch = frame.shape[2] if len(frame.shape) == 3 else 1
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_duration = self.total_frames / self.video_fps if self.video_fps > 0 else 0

        self.logger.info(
            f"✅ 비디오 정보: {w}x{h}x{ch}, {self.video_fps:.1f}fps, {self.video_duration:.1f}초, {self.total_frames}프레임")

        # 처음부터 다시 시작
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.video_capture = cap
        self.current_frame_idx = 0
        return True

    def jump_to_time(self, target_seconds):
        """특정 시간(초)으로 이동"""
        if not self.video_capture or not self.video_capture.isOpened():
            return False

        # 시간을 프레임 인덱스로 변환
        target_frame = int(target_seconds * self.video_fps)
        target_frame = max(0, min(target_frame, self.total_frames - 1))

        try:
            # 프레임 위치 설정
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            self.current_frame_idx = target_frame
            current_time = self.get_current_time()
            self.logger.info(f"⭐ 시간 이동: {current_time:.1f}초 (프레임 {target_frame})")
            return True
        except Exception as e:
            self.logger.error(f"시간 이동 실패: {e}")
            return False

    def jump_relative(self, seconds_delta):
        """현재 위치에서 상대적으로 이동 (앞으로 +, 뒤로 -)"""
        current_time = self.get_current_time()
        target_time = current_time + seconds_delta

        # 범위 제한
        target_time = max(0, min(target_time, self.video_duration))
        return self.jump_to_time(target_time)

    def get_current_time(self):
        """현재 재생 시간(초) 반환"""
        if self.video_fps == 0:
            return 0
        return self.current_frame_idx / self.video_fps

    def get_total_duration(self):
        """전체 영상 길이(초) 반환"""
        return self.video_duration

    def read_frame(self):
        """프레임 읽기 (반복 재생 지원)"""
        if not self.video_capture or not self.video_capture.isOpened():
            return False, None

        ret, frame = self.video_capture.read()
        if ret and frame is not None:
            self.current_frame_idx += 1
            return True, frame
        else:
            # 비디오 끝에 도달 - 반복 재생
            loop_enabled = self.config.get('video_mode', {}).get('loop', True)
            if loop_enabled and self.total_frames > 0:
                self.logger.info("🔄 비디오 반복 재생 시작")
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame_idx = 0
                ret, frame = self.video_capture.read()
                if ret and frame is not None:
                    self.current_frame_idx = 1
                    return True, frame

            return False, None

    def get_fps(self):
        """FPS 반환"""
        return self.video_fps

    def get_progress(self):
        """재생 진행률 반환 (0.0 ~ 1.0)"""
        if self.total_frames == 0:
            return 0.0
        return min(self.current_frame_idx / self.total_frames, 1.0)

    def release(self):
        """리소스 해제"""
        if self.video_capture:
            self.video_capture.release()


class LaneDetector:
    """차선 감지 및 동적 Front Zone 계산"""

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
    """개선된 IoU 기반 객체 추적기 - 앞차 출발 오탐 방지"""

    def __init__(self, max_lost=5, iou_threshold=0.3, movement_threshold=2, expected_fps=30):
        self.tracks = {}
        self.next_id = 1
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.movement_threshold = movement_threshold
        self.frame_count = 0

        # ✅ 새로운 출발 감지 파라미터 (영상 속도 저하 대응)
        self.expected_fps = expected_fps
        self.departure_buffer_frames = int(expected_fps * 2)  # 2초 상당의 프레임
        self.stationary_threshold = 1.5  # 정지 상태 판정 임계값
        self.min_stationary_ratio = 0.7  # 최소 정지 비율 (70%)

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
        """✅ 개선된 출발 감지: 시간 기반 버퍼링으로 오탐 방지"""
        if len(track['movement_history']) < self.departure_buffer_frames:
            return False

        # 최근 N프레임의 움직임 분석
        recent_movements = list(track['movement_history'])[-self.departure_buffer_frames:]

        # 1단계: 정지 상태 판정 (최근 10프레임 제외하고 분석)
        analysis_frames = recent_movements[:-15] if len(recent_movements) > 15 else recent_movements[:-5]
        if not analysis_frames:
            return False

        stationary_count = 0
        for move_x, move_y in analysis_frames:
            distance = (move_x ** 2 + move_y ** 2) ** 0.5
            if distance < self.stationary_threshold:
                stationary_count += 1

        # 2단계: 충분히 오래 정지했는지 확인
        stationary_ratio = stationary_count / len(analysis_frames)
        if stationary_ratio < self.min_stationary_ratio:
            return False  # 충분히 정지하지 않았음

        # 3단계: 최근 프레임에서 실제 움직임 확인
        recent_movement_frames = recent_movements[-10:] if len(recent_movements) >= 10 else recent_movements[-5:]
        avg_recent_distance = sum(
            (move_x ** 2 + move_y ** 2) ** 0.5
            for move_x, move_y in recent_movement_frames
        ) / len(recent_movement_frames)

        # 4단계: 출발 판정 (영상 속도 저하를 고려한 임계값)
        adjusted_threshold = self.movement_threshold * (2.0 if is_in_zone else 3.0)
        is_departed = avg_recent_distance > adjusted_threshold

        # 5단계: 디버그 정보
        if is_departed:
            track['departure_info'] = {
                'stationary_ratio': stationary_ratio,
                'recent_movement': avg_recent_distance,
                'threshold_used': adjusted_threshold,
                'buffer_frames': len(recent_movements)
            }

        return is_departed

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

                # ✅ 개선된 출발 감지 적용
                if self.detect_departure_improved(track, is_in_zone):
                    track['is_moving'] = True
                    track['departure_detected'] = True

                matched_tracks.add(track_id)
                matched_dets.add(best_idx)

        for track_id, track in active_tracks.items():
            if track_id not in matched_tracks:
                track['lost'] += 1

        # ✅ 더 긴 움직임 히스토리 유지 (영상 속도 저하 대응)
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
                    'movement_history': deque(maxlen=max_history),  # 더 긴 히스토리
                    'is_moving': False,
                    'departure_detected': False,
                    'departure_info': {}
                }
                self.next_id += 1

        self.tracks = {tid: tr for tid, tr in self.tracks.items() if tr['lost'] <= self.max_lost}
        return self.get_active_tracks()

    def get_active_tracks(self):
        return [tr for tr in self.tracks.values() if tr['lost'] == 0]


class TrafficLightColorDetector:
    """신호등 색상 감지기 - 앞차 출발 판단용"""

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
        red_lower3 = np.array([0, 80, 80]);
        red_upper3 = np.array([15, 255, 255])
        red_lower4 = np.array([165, 80, 80]);
        red_upper4 = np.array([180, 255, 255])
        yellow_lower = np.array([18, 120, 120]);
        yellow_upper = np.array([35, 255, 255])
        green_lower = np.array([45, 100, 100]);
        green_upper = np.array([90, 255, 255])

        red_mask = (cv2.inRange(hsv, red_lower1, red_upper1) +
                    cv2.inRange(hsv, red_lower2, red_upper2) +
                    cv2.inRange(hsv, red_lower3, red_upper3) +
                    cv2.inRange(hsv, red_lower4, red_upper4))
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


class MP4BlackBoxSystem:
    """MP4 영상 전용 앞차 출발 오탐 방지 블랙박스 시스템 + 시간 탐색 기능"""

    def __init__(self, config_path="mp4_blackbox_config.json"):
        self.setup_logging()
        self.config_path = config_path
        self.config = self.load_config()

        self.video_manager = VideoManager(self.logger, self.config)
        self.time_navigator = TimeNavigator(self.logger)  # ✅ 시간 탐색 기능 추가
        self.net = None
        self.classes = []
        self.output_layers = []

        self.running = False
        self.current_frame = None
        self.video_writer = None
        self.paused = False  # ✅ 일시정지 기능

        # ✅ 개선된 추적기 적용 (영상 FPS 고려)
        expected_fps = 30  # 기본 FPS
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
            'vehicle_departures': 0,
            'false_departures_prevented': 0  # ✅ 오탐 방지 통계
        }

        self.frame_read_failures = 0
        self.max_frame_read_failures = 10

        # ✅ 성능 모니터링 추가
        self.processing_times = deque(maxlen=30)
        self.last_detection_time = 0

        self.logger.info("✅ MP4 전용 앞차 출발 오탐 방지 블랙박스 시스템 + 시간 탐색 초기화 완료")

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
                logging.FileHandler('log/mp4_blackbox.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self):
        default_config = {
            "video": {
                "file_path": "blackbox.mp4",  # ✅ 파일명 적용
                "output_dir": "output"
            },
            "video_mode": {
                "loop": True,
                "speed": 1.0,
                "save_result": True
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
                "detection_interval": 2,  # ✅ 간격 조정으로 성능 향상
                "max_processing_time": 0.1
            },
            "traffic_light": {
                "enable_detection": True,
                "stability_frames": 2
            },
            "tracking": {
                "iou_threshold": 0.25,  # ✅ 더 관대한 추적
                "max_lost_frames": 8,  # ✅ 더 오래 추적
                "movement_threshold": 2,
                "departure_buffer_seconds": 2.5,  # ✅ 출발 판정 버퍼 시간
                "stationary_threshold": 1.5,
                "min_stationary_ratio": 0.7
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

    def init_video(self):
        video_path = self.config['video']['file_path']
        success = self.video_manager.init_video(video_path)

        # ✅ 실제 FPS를 추적기에 반영
        if success:
            actual_fps = self.video_manager.get_fps()
            self.vehicle_tracker.expected_fps = actual_fps
            self.vehicle_tracker.departure_buffer_frames = int(actual_fps * 2.5)  # 2.5초 버퍼
            self.logger.info(
                f"✅ 실제 영상 FPS: {actual_fps:.1f}, 출발 판정 버퍼: {self.vehicle_tracker.departure_buffer_frames}프레임")

        return success

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

    def create_video_writer(self, frame_shape):
        try:
            if not self.config['video_mode'].get('save_result', True):
                return None, None

            output_dir = self.config['video']['output_dir']
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"MP4_arrow_key_fixed_{timestamp}.mp4"  # ✅ 화살표 키 수정 버전 표시
            filepath = os.path.join(output_dir, filename)

            height, width = frame_shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = self.video_manager.get_fps()

            if width % 2 != 0:
                width -= 1
            if height % 2 != 0:
                height -= 1

            writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
            if writer.isOpened():
                self.logger.info(f"✅ 처리 결과 저장 시작: {filename}")
                return writer, filepath
            else:
                self.logger.error(f"❌ 비디오 라이터 생성 실패")
                return None, None
        except Exception as e:
            self.logger.error(f"비디오 라이터 생성 실패: {e}")
            return None, None

    def filter_vehicles_by_lane(self, vehicle_detections, frame_width, frame_height, frame):
        """차선 기반 앞차 필터링 - 모든 이동체 포함"""
        lanes = self.lane_detector.detect_lanes(frame)
        if lanes is None:
            return []

        center_points = self.lane_detector.calculate_lane_center_points(lanes, frame_height)
        if not center_points:
            return []

        filtered = []
        lane_width = 120  # 차선 폭 확대
        min_area = (frame_width * frame_height) * 0.008  # 최소 면적 완화

        for det in vehicle_detections:
            if det['class_name'] not in ['car', 'truck', 'bus', 'motorbike', 'bicycle']:
                continue

            x, y, w, h = det['box']
            vehicle_center_x = x + w / 2
            vehicle_center_y = y + h / 2

            # 앞쪽 차선 영역 확대
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
        """✅ 성능 최적화된 객체 감지"""
        start_time = time.time()

        # 처리 시간 제한으로 성능 보장
        max_processing_time = self.config['detection'].get('max_processing_time', 0.1)

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
            # 처리 시간 체크
            if time.time() - start_time > max_processing_time:
                break

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

        # 처리 시간 기록
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
        """✅ 개선된 출발 로그 (디버그 정보 포함)"""
        if track.get('departure_detected', False):
            log_message = "앞의 차량이 출발하였습니다"

            # 디버그 정보 포함
            departure_info = track.get('departure_info', {})
            detailed_log = (f"🚗 앞차 출발: ID{track['id']} ({track['class_name']}) - "
                            f"정지비율: {departure_info.get('stationary_ratio', 0):.2f}, "
                            f"최근움직임: {departure_info.get('recent_movement', 0):.2f}, "
                            f"임계값: {departure_info.get('threshold_used', 0):.2f}")

            self.logger.info(detailed_log)
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

    def draw_lane_overlay(self, overlay, lanes, center_points, width, height):
        if lanes is None:
            return
        # 차선 라인
        if lanes.get('left'):
            x1, y1, x2, y2 = lanes['left']
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if lanes.get('right'):
            x1, y1, x2, y2 = lanes['right']
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if center_points:
            # 중앙선을 노란색으로
            for i in range(len(center_points) - 1):
                cv2.line(overlay, center_points[i], center_points[i + 1], (0, 255, 255), 3)
            # 차선 폭 표시
            lane_width = 120
            for point in center_points:
                cx, cy = point
                cv2.rectangle(overlay, (cx - lane_width // 2, cy - 1), (cx + lane_width // 2, cy + 1),
                              (0, 255, 255), -1)

    def draw_time_navigation_overlay(self, overlay, width, height):
        """✅ 개선된 시간 탐색 UI 오버레이 - 더 명확한 키 안내"""
        if not self.time_navigator.should_show_ui():
            return

        # 반투명 배경
        nav_overlay = overlay.copy()
        nav_height = 160  # 높이 증가
        nav_y = height - nav_height - 50
        cv2.rectangle(nav_overlay, (50, nav_y), (width - 50, nav_y + nav_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, nav_overlay, 0.3, 0, overlay)

        # 테두리
        cv2.rectangle(overlay, (50, nav_y), (width - 50, nav_y + nav_height), (0, 255, 255), 2)

        # 제목
        cv2.putText(overlay, "TIME NAVIGATION CONTROLS", (60, nav_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 현재 시간 정보
        current_time = self.video_manager.get_current_time()
        total_time = self.video_manager.get_total_duration()
        progress = self.video_manager.get_progress()
        time_text = f"Time: {self.time_navigator.format_time(current_time)} / {self.time_navigator.format_time(total_time)} ({progress * 100:.1f}%)"
        cv2.putText(overlay, time_text, (60, nav_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 점프 시간 설정 표시
        jump_seconds = self.time_navigator.get_current_jump_seconds()
        jump_text = f"Jump Time: {jump_seconds}s"
        cv2.putText(overlay, jump_text, (60, nav_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 키보드 컨트롤 안내 (더 상세히)
        controls_text = [
            "[ARROW KEYS] Jump [A/D] 1s [1-6] Set jump time [SPACE] Pause",
            "[T] Cycle jump [Home/End] Start/End [+/-] Speed [H] Help [Q] Quit"
        ]

        for i, text in enumerate(controls_text):
            cv2.putText(overlay, text, (60, nav_y + 100 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 일시정지 표시
        if self.paused:
            cv2.putText(overlay, "PAUSED", (width - 150, nav_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 현재 재생 속도 표시
        speed = self.config['video_mode'].get('speed', 1.0)
        if speed != 1.0:
            speed_text = f"Speed: {speed:.1f}x"
            cv2.putText(overlay, speed_text, (width - 150, nav_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    def draw_permanent_controls_hint(self, overlay, width, height):
        """화면 하단에 항상 표시되는 기본 컨트롤 힌트"""
        hint_text = "CONTROLS: [ARROWS] Navigate [SPACE] Pause [T] Jump time [H] Help [Q] Quit"

        # 텍스트 크기 계산
        (text_w, text_h), _ = cv2.getTextSize(hint_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # 반투명 배경
        hint_y = height - 25
        cv2.rectangle(overlay, (5, hint_y - text_h - 5), (text_w + 15, height - 5), (0, 0, 0), -1)

        # 텍스트 표시
        cv2.putText(overlay, hint_text, (10, hint_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_optimized_overlay(self, frame, detections, tracked_vehicles):
        """✅ 개선된 오버레이 - 항상 표시되는 컨트롤 힌트 추가"""
        overlay = frame.copy()
        height, width = frame.shape[:2]

        # 차선 오버레이
        lanes = self.lane_detector.last_valid_lanes
        center_points = self.lane_detector.calculate_lane_center_points(lanes, height) if lanes else None
        self.draw_lane_overlay(overlay, lanes, center_points, width, height)

        # 검출된 차량들 표시
        for detection in detections:
            x, y, w, h = detection['box']
            class_name = detection['class_name']
            confidence = detection['confidence']

            if class_name in ['car', 'truck', 'bus', 'motorbike', 'bicycle']:
                vehicle_center_x = x + w / 2
                vehicle_center_y = y + h / 2

                if vehicle_center_y > height * 0.6:
                    is_in_lane = False
                    if center_points:
                        for cx, cy in center_points:
                            if abs(vehicle_center_y - cy) < 30:
                                lane_width = 120
                                if abs(vehicle_center_x - cx) < lane_width // 2:
                                    is_in_lane = True
                                    break

                    if is_in_lane:
                        colors = {
                            'car': (0, 255, 0),
                            'truck': (0, 255, 255),
                            'bus': (255, 255, 0),
                            'motorbike': (255, 0, 255),
                            'bicycle': (100, 255, 255)
                        }
                        color = colors.get(class_name, (0, 255, 0))
                        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 3)

                        conf_text = f"{class_name}: {confidence:.2f}"
                        cv2.putText(overlay, conf_text, (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            elif class_name == 'traffic light':
                color = (255, 255, 255)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 3)

                current_color = self.traffic_light_detector.last_stable_color
                if current_color:
                    signal_colors = {'red': (0, 0, 255), 'yellow': (0, 255, 255), 'green': (0, 255, 0)}
                    signal_color = signal_colors.get(current_color, (255, 255, 255))
                    cv2.circle(overlay, (x + w // 2, y - 20), 10, signal_color, -1)
                    cv2.circle(overlay, (x + w // 2, y - 20), 10, (255, 255, 255), 2)

        # 추적된 앞차 표시
        for tr in tracked_vehicles:
            x, y, w, h = tr['bbox']
            track_id = tr['id']
            class_name = tr['class_name']

            if tr.get('departure_detected', False):
                track_color = (0, 255, 255)
                status_text = "DEPARTED"
            elif tr.get('is_moving', False):
                track_color = (255, 255, 0)
                status_text = "MOVING"
            else:
                track_color = (0, 255, 0)
                status_text = "WAITING"

            cv2.rectangle(overlay, (x, y), (x + w, y + h), track_color, 4)

            id_text = f"FRONT-{track_id} ({status_text})"
            (text_w, text_h), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(overlay, (x + 5, y + 5), (x + text_w + 15, y + text_h + 15), (0, 0, 0), -1)
            cv2.putText(overlay, id_text, (x + 10, y + text_h + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, track_color, 2)

            history_len = len(tr.get('movement_history', []))
            buffer_needed = self.vehicle_tracker.departure_buffer_frames
            history_text = f"H:{history_len}/{buffer_needed}"
            cv2.putText(overlay, history_text, (x + 10, y + h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, track_color, 1)

        # 상단 패널
        panel_h = 50
        panel_overlay = overlay.copy()
        cv2.rectangle(panel_overlay, (0, 0), (width, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, panel_overlay, 0.2, 0, overlay)
        cv2.line(overlay, (0, panel_h), (width, panel_h), (0, 255, 0), 2)

        # 시간 정보
        progress = self.video_manager.get_progress()
        current_time = self.video_manager.get_current_time()
        total_time = self.video_manager.get_total_duration()
        time_text = f"{self.time_navigator.format_time(current_time)} / {self.time_navigator.format_time(total_time)} ({progress * 100:.1f}%)"
        cv2.putText(overlay, time_text, (15, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 성능 정보
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        perf_text = f"PROC: {avg_processing_time * 1000:.1f}ms"
        cv2.putText(overlay, perf_text, (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 앞차 감지 상태
        waiting_vehicles = len([t for t in tracked_vehicles if not t.get('is_moving', False)])
        moving_vehicles = len([t for t in tracked_vehicles if t.get('is_moving', False)])
        departed_vehicles = len([t for t in tracked_vehicles if t.get('departure_detected', False)])

        status_text = f"WAIT:{waiting_vehicles} MOVE:{moving_vehicles} DEPT:{departed_vehicles}"
        cv2.putText(overlay, status_text, (300, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 재생 속도 표시
        speed = self.config['video_mode'].get('speed', 1.0)
        speed_text = f"Speed: {speed:.1f}x"
        cv2.putText(overlay, speed_text, (300, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # 신호등 상태
        if self.traffic_light_detector.last_stable_color:
            signal_text = f"SIGNAL: {self.traffic_light_detector.last_stable_color.upper()}"
            cv2.putText(overlay, signal_text, (width - 200, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 일시정지 상태
        if self.paused:
            cv2.putText(overlay, "PAUSED", (width - 200, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 시간 탐색 UI (필요할 때만)
        self.draw_time_navigation_overlay(overlay, width, height)

        # ✅ 항상 표시되는 컨트롤 힌트 추가
        self.draw_permanent_controls_hint(overlay, width, height)

        return overlay

    def handle_keyboard_input(self, key):
        """✅ waitKeyEx()를 사용한 완전한 키보드 입력 처리 - 모든 플랫폼 지원"""

        # 키가 눌리지 않은 경우 (-1)는 무시
        if key == -1:
            return False

        # 디버깅용: 키 코드 확인 (필요시 주석 해제)
        # print(f"Key pressed: {key} (0x{key:X})")

        # 기본 문자 키들
        if key == ord('q') or key == ord('Q'):
            self.logger.info("사용자 요청으로 종료합니다.")
            self.running = False
            return True

        elif key == ord('s') or key == ord('S'):
            screenshot_name = f"arrow_key_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            if self.current_frame is not None:
                cv2.imwrite(screenshot_name, self.current_frame)
                self.logger.info(f"스크린샷 저장: {screenshot_name}")

        elif key == ord('t') or key == ord('T'):
            # 점프 시간 순환
            jump_time = self.time_navigator.cycle_jump_time()

        elif key == 32:  # SPACE - 일시정지/재개
            self.paused = not self.paused
            status = "일시정지" if self.paused else "재개"
            self.logger.info(f"⏸️ 영상 {status}")

        # ✅ 화살표 키 처리 - Windows, Linux, macOS 모든 플랫폼 지원
        elif key in [2424832, 65361, 81]:  # LEFT ARROW (Windows, Linux, macOS)
            jump_seconds = self.time_navigator.get_current_jump_seconds()
            if self.video_manager.jump_relative(-jump_seconds):
                self.time_navigator.show_navigation_ui = True
                self.time_navigator.ui_show_time = time.time()
                self.logger.info(f"⬅️ {jump_seconds}초 뒤로 점프")

        elif key in [2555904, 65363, 83]:  # RIGHT ARROW (Windows, Linux, macOS)
            jump_seconds = self.time_navigator.get_current_jump_seconds()
            if self.video_manager.jump_relative(jump_seconds):
                self.time_navigator.show_navigation_ui = True
                self.time_navigator.ui_show_time = time.time()
                self.logger.info(f"➡️ {jump_seconds}초 앞으로 점프")

        elif key in [2490368, 65362, 82]:  # UP ARROW (Windows, Linux, macOS)
            if self.video_manager.jump_relative(10):
                self.time_navigator.show_navigation_ui = True
                self.time_navigator.ui_show_time = time.time()
                self.logger.info("⬆️ 10초 앞으로 이동")

        elif key in [2621440, 65364, 84]:  # DOWN ARROW (Windows, Linux, macOS)
            if self.video_manager.jump_relative(-10):
                self.time_navigator.show_navigation_ui = True
                self.time_navigator.ui_show_time = time.time()
                self.logger.info("⬇️ 10초 뒤로 이동")

        elif key in [2359296, 65360, 71]:  # HOME (Windows, Linux, macOS)
            if self.video_manager.jump_to_time(0):
                self.time_navigator.show_navigation_ui = True
                self.time_navigator.ui_show_time = time.time()
                self.logger.info("🏠 영상 시작으로 이동")

        elif key in [2293760, 65367, 79]:  # END (Windows, Linux, macOS)
            total_time = self.video_manager.get_total_duration()
            target_time = max(0, total_time - 10)
            if self.video_manager.jump_to_time(target_time):
                self.time_navigator.show_navigation_ui = True
                self.time_navigator.ui_show_time = time.time()
                self.logger.info("🔚 영상 끝으로 이동")

        # 숫자 키로 직접 점프 시간 설정
        elif key == ord('1'):
            self.time_navigator.current_jump_index = 0  # 5초
            self.time_navigator.show_navigation_ui = True
            self.time_navigator.ui_show_time = time.time()
            self.logger.info(f"⏱️ 점프 시간: {self.time_navigator.get_current_jump_seconds()}초")

        elif key == ord('2'):
            self.time_navigator.current_jump_index = 1  # 10초
            self.time_navigator.show_navigation_ui = True
            self.time_navigator.ui_show_time = time.time()
            self.logger.info(f"⏱️ 점프 시간: {self.time_navigator.get_current_jump_seconds()}초")

        elif key == ord('3'):
            self.time_navigator.current_jump_index = 2  # 30초
            self.time_navigator.show_navigation_ui = True
            self.time_navigator.ui_show_time = time.time()
            self.logger.info(f"⏱️ 점프 시간: {self.time_navigator.get_current_jump_seconds()}초")

        elif key == ord('4'):
            self.time_navigator.current_jump_index = 3  # 60초
            self.time_navigator.show_navigation_ui = True
            self.time_navigator.ui_show_time = time.time()
            self.logger.info(f"⏱️ 점프 시간: {self.time_navigator.get_current_jump_seconds()}초")

        elif key == ord('5'):
            self.time_navigator.current_jump_index = 4  # 120초
            self.time_navigator.show_navigation_ui = True
            self.time_navigator.ui_show_time = time.time()
            self.logger.info(f"⏱️ 점프 시간: {self.time_navigator.get_current_jump_seconds()}초")

        elif key == ord('6'):
            self.time_navigator.current_jump_index = 5  # 300초
            self.time_navigator.show_navigation_ui = True
            self.time_navigator.ui_show_time = time.time()
            self.logger.info(f"⏱️ 점프 시간: {self.time_navigator.get_current_jump_seconds()}초")

        # A/D 키로 미세 조정
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

        # 재생 속도 조절
        elif key == ord('+') or key == ord('='):
            current_speed = self.config['video_mode'].get('speed', 1.0)
            new_speed = min(current_speed * 1.5, 4.0)
            self.config['video_mode']['speed'] = new_speed
            self.logger.info(f"⚡ 재생 속도: {new_speed:.1f}x")

        elif key == ord('-') or key == ord('_'):
            current_speed = self.config['video_mode'].get('speed', 1.0)
            new_speed = max(current_speed / 1.5, 0.25)
            self.config['video_mode']['speed'] = new_speed
            self.logger.info(f"🐌 재생 속도: {new_speed:.1f}x")

        # 도움말 표시
        elif key == ord('h') or key == ord('H'):
            self.show_help()

        return False  # 종료하지 않음

    def show_help(self):
        """키보드 컨트롤 도움말 표시"""
        help_text = """
==================== 키보드 컨트롤 ====================
[기본 제어]
Q - 프로그램 종료
SPACE - 일시정지/재개
S - 스크린샷 저장
H - 이 도움말 표시

[시간 탐색] ✅ 화살표 키 완전 수정됨
←→ - 설정된 시간만큼 점프 (기본 30초)
↑↓ - 10초 점프
A/D - 1초 미세 조정
Home - 영상 시작으로
End - 영상 끝으로

[점프 시간 설정]
T - 점프 시간 순환 (5→10→30→60→120→300초)
1 - 5초 점프로 설정
2 - 10초 점프로 설정
3 - 30초 점프로 설정
4 - 60초 점프로 설정
5 - 120초 점프로 설정
6 - 300초 점프로 설정

[재생 속도]
+/= - 재생 속도 증가 (최대 4배속)
-/_ - 재생 속도 감소 (최소 0.25배속)
====================================================
"""
        self.logger.info(help_text)
        print(help_text)

    def signal_handler(self, signum, frame):
        self.running = False
        self.cleanup_resources()
        sys.exit(0)

    def cleanup_resources(self):
        try:
            if hasattr(self, 'video_writer') and self.video_writer:
                self.video_writer.release()
            self.video_manager.release()
            cv2.destroyAllWindows()
            if hasattr(self, 'tts_system') and self.tts_system:
                self.tts_system.shutdown()
        except Exception as e:
            self.logger.error(f"리소스 정리 실패: {e}")

    def run(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        if not self.init_video():
            return False

        ai_ok = self.init_ai_model()
        if not ai_ok:
            self.logger.warning("AI 모델 없이 기본 처리 모드로 실행")

        self.running = True
        frame_count = 0
        last_fps_time = time.time()
        fps_counter = 0

        # 첫 프레임으로 Writer 준비
        ok, first = self.video_manager.read_frame()
        if not ok or first is None:
            self.logger.error("초기 프레임을 읽을 수 없습니다.")
            self.cleanup_resources()
            return False

        self.video_writer, video_path = self.create_video_writer(first.shape)

        # 미리보기 창 설정
        if self.config['display'].get('show_preview', True):
            cv2.namedWindow("MP4 Arrow Key Fixed Player", cv2.WINDOW_NORMAL)
            window_w = self.config['display'].get('window_width', 1280)
            window_h = self.config['display'].get('window_height', 720)
            cv2.resizeWindow("MP4 Arrow Key Fixed Player", window_w, window_h)

        # 비디오 재생 속도 조절
        video_fps = self.video_manager.get_fps()
        frame_delay = 1.0 / video_fps if video_fps > 0 else 1.0 / 30
        speed_multiplier = self.config['video_mode'].get('speed', 1.0)
        frame_delay = frame_delay / speed_multiplier

        total_duration = self.video_manager.get_total_duration()
        self.logger.info(
            f"🎬 MP4 화살표 키 수정 완료 + 앞차 감지 시작 (FPS: {video_fps:.1f}, 길이: {self.time_navigator.format_time(total_duration)}, 속도: {speed_multiplier}x)")
        self.logger.info("✅ 화살표 키 작동 확인: [←→]점프 [↑↓]±10초 [T]점프설정 [Space]일시정지 [Q]종료")

        while self.running:
            try:
                if not self.paused:
                    ret, frame = self.video_manager.read_frame()
                    if not ret or frame is None:
                        if self.config['video_mode'].get('loop', True):
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

                    overlay_frame = self.draw_optimized_overlay(frame.copy(), detections, tracked_vehicles)

                    # 앞차 출발 감지 로그
                    for track in tracked_vehicles:
                        if track.get('departure_detected', False) and not track.get('logged', False):
                            self.log_vehicle_departure(track)
                            track['logged'] = True

                    # 결과 비디오 저장
                    if self.video_writer and self.video_writer.isOpened():
                        try:
                            h, w = overlay_frame.shape[:2]
                            if w % 2 != 0:
                                overlay_frame = overlay_frame[:, :-1]
                            if h % 2 != 0:
                                overlay_frame = overlay_frame[:-1, :]
                            self.video_writer.write(overlay_frame)
                        except Exception as e:
                            self.logger.error(f"프레임 저장 실패: {e}")

                    frame_count += 1

                else:
                    # 일시정지 상태에서는 현재 프레임을 그대로 사용
                    if self.current_frame is not None:
                        overlay_frame = self.draw_optimized_overlay(self.current_frame.copy(), [], [])
                    else:
                        time.sleep(0.1)
                        continue

                # ✅ 실시간 미리보기 창 표시 - waitKeyEx() 사용
                if self.config['display'].get('show_preview', True):
                    cv2.imshow("MP4 Arrow Key Fixed Player", overlay_frame)

                    # ✅ waitKeyEx()로 변경하여 화살표 키 완전 지원
                    key = cv2.waitKeyEx(1)

                    if self.handle_keyboard_input(key):
                        break

                # FPS 및 통계 (개선된 정보)
                if not self.paused:
                    fps_counter += 1
                    if fps_counter % 30 == 0:
                        current_time = time.time()
                        fps = 30 / (current_time - last_fps_time) if (current_time - last_fps_time) > 0 else 0
                        last_fps_time = current_time
                        active_tracks = len([])  # tracked_vehicles는 일시정지시에도 표시되므로
                        waiting_vehicles = 0
                        moving_vehicles = 0
                        progress = self.video_manager.get_progress()
                        avg_proc_time = sum(self.processing_times) / len(
                            self.processing_times) if self.processing_times else 0
                        current_video_time = self.video_manager.get_current_time()

                        self.logger.info(
                            f"📊 시간: {self.time_navigator.format_time(current_video_time)} | 진행률: {progress * 100:.1f}% | FPS: {fps:.1f} | "
                            f"처리시간: {avg_proc_time * 1000:.1f}ms | 출발: {self.detection_stats['vehicle_departures']} | "
                            f"신호등: {self.detection_stats['traffic_lights']}, 변화: {self.detection_stats['traffic_light_changes']}"
                        )

                    # 재생 속도 조절 (일시정지가 아닐 때만)
                    time.sleep(frame_delay)

            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt 감지")
                break
            except Exception as e:
                self.logger.error(f"실행 중 오류: {e}")
                break

        self.cleanup_resources()
        return True


def main():
    parser = argparse.ArgumentParser(description='MP4 전용 화살표 키 완전 수정 블랙박스 시스템')
    parser.add_argument('--config', default='mp4_blackbox_config.json', help='설정 파일 경로')
    parser.add_argument('--video', help='비디오 파일 경로 (설정 파일보다 우선)')
    parser.add_argument('--debug', action='store_true', help='디버그 모드')
    args = parser.parse_args()

    try:
        blackbox = MP4BlackBoxSystem(args.config)

        # 명령행에서 비디오 파일이 지정된 경우
        if args.video:
            blackbox.config['video']['file_path'] = args.video

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
