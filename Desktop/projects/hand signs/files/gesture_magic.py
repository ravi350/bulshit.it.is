#!/usr/bin/env python3
"""
GestureMagic — Python Edition
Real-time hand gesture recognition using MediaPipe + OpenCV
Detects gestures and renders animated visual effects directly on frame.

Requirements:
    pip install opencv-python mediapipe numpy

Usage:
    python gesture_magic.py
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ══════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════

WINDOW_NAME    = "GestureMagic — Press Q to quit"
CAMERA_INDEX   = 0
FRAME_WIDTH    = 1280
FRAME_HEIGHT   = 720
SWIPE_FRAMES   = 15         # frames of history for swipe detection
SWIPE_THRESHOLD = 0.10      # fraction of frame width

# Gesture → (color BGR, label)
GESTURE_MAP = {
    "open_palm":   ((255, 140, 50),  "Open Palm  ⚡ ENERGY WAVE"),
    "fist":        ((60,  60,  240), "Fist       💥 SHOCKWAVE"),
    "thumbs_up":   ((50,  200, 80),  "Thumbs Up  ✨ AURA"),
    "pointing":    ((50,  200, 240), "Pointing   🔦 LASER"),
    "peace":       ((220, 80,  240), "Peace ✌    🌈 PRISM BURST"),
    "pinch":       ((240, 200, 50),  "Pinch      🌀 VORTEX"),
    "ok_sign":     ((100, 220, 50),  "OK Sign    ○  RESONANCE"),
    "swipe_right": ((60,  140, 255), "Swipe →    ➡ SLASH TRAIL"),
    "swipe_left":  ((240, 80,  200), "Swipe ←    ⬅ SLASH TRAIL"),
    "rock":        ((50,  240, 220), "Rock On 🤘 ⚡ LIGHTNING"),
}


# ══════════════════════════════════════════════════════════════════
#  Particle / Effect Data Classes
# ══════════════════════════════════════════════════════════════════

@dataclass
class Particle:
    x: float; y: float
    vx: float; vy: float
    life: float; decay: float
    size: float
    color: Tuple[int, int, int]
    kind: str = "dot"
    gravity: float = 0.0
    angle: float = 0.0
    cx: float = 0.0; cy: float = 0.0
    radius: float = 0.0
    angular_speed: float = 0.0
    in_speed: float = 0.0


@dataclass
class Effect:
    kind: str
    x: float; y: float
    color: Tuple[int,int,int]
    life: float; decay: float
    # Shockwave
    r: float = 0; max_r: float = 200; width: int = 3
    # Slash
    x1: float = 0; y1: float = 0; x2: float = 0; y2: float = 0
    progress: float = 0; speed: float = 0.05
    # Lightning
    branches: list = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════
#  GestureMagic Engine
# ══════════════════════════════════════════════════════════════════

class GestureMagic:
    def __init__(self):
        # MediaPipe Tasks API
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        BaseOptions = python.BaseOptions
        HandLandmarker = vision.HandLandmarker
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        self.model_path = "hand_landmarker.task"
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        # Camera
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        # State
        self.particles: List[Particle] = []
        self.effects:   List[Effect]   = []
        # Note: now per-hand lists initialized in __init__
        self.swipe_histories = [deque(maxlen=SWIPE_FRAMES) for _ in range(2)]
        self.gesture_label_text  = ""
        self.gesture_label_until = 0.0
        self.finger_states = [[False] * 5 for _ in range(2)]
        self.current_gestures = [None, None]
        self.prev_gestures = [None, None]

        # FPS tracking
        self.fps_time   = time.time()
        self.fps_frames = 0
        self.fps        = 0

    # ── Main Loop ────────────────────────────────────────────────

    def run(self):
        print("GestureMagic started. Press Q to quit, S to toggle skeleton.")
        show_skeleton = True

        while self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)   # Mirror
            H, W = frame.shape[:2]

            # ── MediaPipe Inference ─────────────────────────────
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp_ms = int(time.time() * 1000)
            results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            # Dark overlay for cinematic look
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (W,H), (0,0,0), -1)
            frame = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)

            gestures = [None, None]
            if results.hand_landmarks:
                for hand_num, hand_landmarks in enumerate(results.hand_landmarks[:2]):
                    lm_list = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]

                    # Skeleton
                    if show_skeleton:
                        color = GESTURE_MAP.get(self.current_gestures[hand_num], ((100,150,255),""))[0] if self.current_gestures[hand_num] else (100,150,255)
                        self._draw_skeleton(frame, lm_list, W, H, color)

                    # Palm center (landmark 9)
                    palm_x = lm_list[9][0]
                    palm_y = lm_list[9][1]
                    self.swipe_histories[hand_num].append(palm_x)

                    palm_y = lm_list[9][1]  # unused

                    # Finger detection
                    self.finger_states[hand_num] = self._detect_fingers(lm_list)

                    # Gesture classification
                    gesture = self._classify(lm_list, self.finger_states[hand_num], W, self.swipe_histories[hand_num])

                    # FX on gesture change
                    if gesture != self.prev_gestures[hand_num]:
                        if gesture and gesture != "unknown":
                            self._trigger_fx(gesture, lm_list, W, H)
                            self._show_label(gesture)
                        self.prev_gestures[hand_num] = gesture
                    self.current_gestures[hand_num] = gesture
                    gestures[hand_num] = gesture
                    if gesture == "rock":
                        break

            gesture = gestures[0] or gestures[1]

            # ── Render Particles & Effects ──────────────────────
            self._update_particles(frame, W, H)
            self._update_effects(frame, W, H)

            # ── HUD Overlay ─────────────────────────────────────
            self._draw_hud(frame, W, H, gesture, show_skeleton)

            # ── FPS ─────────────────────────────────────────────
            self.fps_frames += 1
            now = time.time()
            if now - self.fps_time >= 1.0:
                self.fps = self.fps_frames
                self.fps_frames = 0
                self.fps_time = now

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('s'): show_skeleton = not show_skeleton

        self.cap.release()
        cv2.destroyAllWindows()

    # ── Skeleton Drawing ─────────────────────────────────────────

    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17)
    ]
    def _draw_skeleton(self, frame, lm_list, W, H, color):
        pts = [(int(x * W), int(y * H)) for x, y, z in lm_list]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for i, (x, y) in enumerate(pts):
            is_tip = i in [4,8,12,16,20]
            r = 6 if is_tip else 4
            cv2.circle(frame, (x, y), r+2, (0,0,0), -1)
            cv2.circle(frame, (x, y), r, color, -1, cv2.LINE_AA)
            if is_tip:
                cv2.circle(frame, (x, y), r+3, color, 1, cv2.LINE_AA)

        # Connections
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)

        # Landmarks
        for i, (x, y) in enumerate(pts):
            is_tip = i in [4, 8, 12, 16, 20]
            r = 6 if is_tip else 4
            cv2.circle(frame, (x, y), r+2, (0,0,0), -1)
            cv2.circle(frame, (x, y), r, color, -1, cv2.LINE_AA)
            if is_tip:
                cv2.circle(frame, (x, y), r+3, color, 1, cv2.LINE_AA)

    # ── Finger Detection ─────────────────────────────────────────

    def _detect_fingers(self, lm):
        # Thumb: tip.x vs ip.x (mirrored frame)
        thumb = lm[4][0] > lm[3][0]
        # Other fingers: tip.y < pip.y
        index  = lm[8][1]  < lm[6][1]
        middle = lm[12][1] < lm[10][1]
        ring   = lm[16][1] < lm[14][1]
        pinky  = lm[20][1] < lm[18][1]
        return [thumb, index, middle, ring, pinky]

    # ── Gesture Classification ────────────────────────────────────

    def _classify(self, lm, fingers, W, swipe_history):
        thumb, index, middle, ring, pinky = fingers
        up_count = sum(fingers)

        # Swipe
        if len(swipe_history) == SWIPE_FRAMES:
            delta = swipe_history[-1] - swipe_history[0]
            if abs(delta) > SWIPE_THRESHOLD and up_count >= 2:
                return "swipe_right" if delta < 0 else "swipe_left"  # Mirrored

        # Pinch: thumb-tip ↔ index-tip
        pinch_dist = math.dist(lm[4][:2], lm[8][:2])
        if pinch_dist < 0.06 and not middle and not ring and not pinky:
            return "pinch"

        # OK: thumb+index pinched, others open
        if pinch_dist < 0.07 and middle and ring and pinky:
            return "ok_sign"

        if up_count == 5: return "open_palm"
        if up_count == 0: return "fist"
        if thumb and not index and not middle and not ring and not pinky: return "thumbs_up"
        if not thumb and index and not middle and not ring and not pinky: return "pointing"
        if not thumb and index and middle and not ring and not pinky:     return "peace"
        if not thumb and index and not middle and not ring and pinky:     return "rock"

        return "unknown"

    # ── FX Triggers ──────────────────────────────────────────────

    def _trigger_fx(self, gesture, lm, W, H):
        cx = int(lm[9][0] * W)
        cy = int(lm[9][1] * H)
        color = GESTURE_MAP.get(gesture, ((100,150,255),""))[0]

        if gesture == "open_palm":
            for _ in range(60):
                angle = random.uniform(0, 2*math.pi)
                speed = random.uniform(4, 10)
                self.particles.append(Particle(
                    x=cx, y=cy,
                    vx=math.cos(angle)*speed, vy=math.sin(angle)*speed,
                    life=1.0, decay=0.02, size=random.uniform(4,8),
                    color=color, kind="energy",
                ))
            self.effects.append(Effect("shockwave", cx, cy, color, 1.0, 0.025, max_r=300, width=3))

        elif gesture == "fist":
            for _ in range(40):
                angle = random.uniform(0, 2*math.pi)
                speed = random.uniform(8, 16)
                self.particles.append(Particle(
                    x=cx, y=cy,
                    vx=math.cos(angle)*speed, vy=math.sin(angle)*speed - 4,
                    life=1.0, decay=0.04, size=random.uniform(3,6),
                    color=color, kind="spark", gravity=0.5,
                ))
            for k in range(3):
                self.effects.append(Effect("shockwave", cx, cy, color, 1.0, 0.03+k*0.01,
                                           r=k*30, max_r=200+k*60, width=4))

        elif gesture == "thumbs_up":
            for _ in range(50):
                self.particles.append(Particle(
                    x=cx + random.uniform(-50,50), y=cy,
                    vx=random.uniform(-1.5,1.5), vy=random.uniform(-6,-2),
                    life=1.0, decay=0.015, size=random.uniform(6,12),
                    color=color, kind="aura",
                ))
            self.effects.append(Effect("shockwave", cx, cy, color, 1.0, 0.025, max_r=150, width=2))

        elif gesture == "pointing":
            tip_x = int(lm[8][0] * W)
            tip_y = int(lm[8][1] * H)
            self.effects.append(Effect("laser", tip_x, tip_y, color, 1.0, 0.04,
                                       x1=tip_x, y1=tip_y, x2=tip_x, y2=0))
            for _ in range(20):
                self.particles.append(Particle(
                    x=tip_x+random.uniform(-15,15), y=tip_y+random.uniform(-15,15),
                    vx=random.uniform(-4,4), vy=random.uniform(-8,-2),
                    life=1.0, decay=0.05, size=random.uniform(2,5),
                    color=color, kind="spark", gravity=0.2,
                ))

        elif gesture == "peace":
            rainbow = [(240,60,60),(240,160,60),(60,240,60),(60,60,240),(200,60,240),(240,60,160)]
            for ci, col in enumerate(rainbow):
                for _ in range(10):
                    angle = (ci / len(rainbow)) * 2*math.pi + random.uniform(-0.3,0.3)
                    speed = random.uniform(5,9)
                    self.particles.append(Particle(
                        x=cx, y=cy,
                        vx=math.cos(angle)*speed, vy=math.sin(angle)*speed,
                        life=1.0, decay=0.018, size=random.uniform(4,8),
                        color=col, kind="energy",
                    ))
            self.effects.append(Effect("rainbow_ring", cx, cy, color, 1.0, 0.02, max_r=200, width=4))

        elif gesture == "pinch":
            for _ in range(50):
                angle = random.uniform(0, 2*math.pi)
                radius = random.uniform(80, 160)
                self.particles.append(Particle(
                    x=cx+math.cos(angle)*radius, y=cy+math.sin(angle)*radius,
                    vx=0, vy=0,
                    life=1.0, decay=0.02, size=random.uniform(3,7),
                    color=color, kind="vortex",
                    cx=cx, cy=cy, angle=angle, radius=radius,
                    angular_speed=random.uniform(0.1,0.15),
                    in_speed=random.uniform(1.5,3.0),
                ))

        elif gesture == "ok_sign":
            for k in range(5):
                self.effects.append(Effect("shockwave", cx, cy, color, 1.0, 0.025+k*0.005,
                                           r=k*20, max_r=150+k*20, width=2))

        elif gesture in ("swipe_right", "swipe_left"):
            going_right = gesture == "swipe_right"
            self.effects.append(Effect(
                "slash", cx, cy, color, 1.0, 0.025,
                x1=(0 if going_right else W), y1=cy - random.randint(0,60),
                x2=(W if going_right else 0), y2=cy + random.randint(0,60),
                speed=0.07,
            ))

        elif gesture == "rock":
            branches = self._gen_lightning(cx, cy, cx+random.randint(-50,50), 0, 6)
            self.effects.append(Effect("lightning", cx, cy, color, 1.0, 0.07,
                                       branches=branches))
            for _ in range(30):
                self.particles.append(Particle(
                    x=cx, y=cy,
                    vx=random.uniform(-8,8), vy=random.uniform(-12,-2),
                    life=1.0, decay=0.05, size=random.uniform(2,5),
                    color=color, kind="spark", gravity=0.4,
                ))

    # ── Lightning Generator ───────────────────────────────────────

    def _gen_lightning(self, x1, y1, x2, y2, depth):
        if depth == 0:
            return [(x1, y1, x2, y2)]
        if depth > 4:
            return []
        mx = (x1+x2)//2 + random.randint(-30, 30)
        my = (y1+y2)//2 + random.randint(-15, 15)
        segs = self._gen_lightning(x1,y1,mx,my,depth-1) + self._gen_lightning(mx,my,x2,y2,depth-1)
        if random.random() > 0.6 and depth > 1:
            bx = mx + random.randint(-40,40)
            by = my + random.randint(10,40)
            segs += self._gen_lightning(mx, my, bx, by, depth-2)
        return segs

    # ── Particle Updater ──────────────────────────────────────────

    def _update_particles(self, frame, W, H):
        alive = []
        for p in self.particles:
            if p.life <= 0: continue
            alpha = p.life
            x, y  = int(p.x), int(p.y)
            sz    = max(1, int(p.size * p.life))
            col   = tuple(int(c * alpha) for c in p.color)

            if p.kind in ("energy", "aura"):
                cv2.circle(frame, (x, y), sz, col, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), sz+2, (int(col[0]*0.5),int(col[1]*0.5),int(col[2]*0.5)), 1, cv2.LINE_AA)
                p.x += p.vx; p.y += p.vy
                p.vx *= 0.96; p.vy *= 0.96

            elif p.kind == "spark":
                x2 = int(p.x - p.vx*2); y2 = int(p.y - p.vy*2)
                cv2.line(frame, (x,y), (x2,y2), col, max(1,sz//2), cv2.LINE_AA)
                p.x += p.vx; p.y += p.vy
                p.vy += p.gravity

            elif p.kind == "vortex":
                p.angle += p.angular_speed
                p.radius -= p.in_speed
                p.x = p.cx + math.cos(p.angle) * p.radius
                p.y = p.cy + math.sin(p.angle) * p.radius
                if p.radius < 5: p.life = 0; continue
                cv2.circle(frame, (int(p.x), int(p.y)), sz, col, -1, cv2.LINE_AA)

            p.life -= p.decay
            alive.append(p)
        self.particles = alive

    # ── Effect Updater ────────────────────────────────────────────

    def _update_effects(self, frame, W, H):
        alive = []
        for e in self.effects:
            if e.life <= 0: continue
            alpha = e.life
            col   = tuple(int(c * alpha) for c in e.color)

            if e.kind == "shockwave":
                r = int(e.r)
                if r > 0:
                    cv2.circle(frame, (int(e.x), int(e.y)), r, col, max(1,int(e.width*e.life)), cv2.LINE_AA)
                e.r += (e.max_r - e.r) * 0.08 + 2

            elif e.kind == "laser":
                tip = (int(e.x), int(e.y))
                end = (int(e.x2), int(e.y2))
                cv2.line(frame, tip, end, col, max(1, int(e.width * e.life)), cv2.LINE_AA)
                cv2.line(frame, tip, end, (255,255,255), 1, cv2.LINE_AA)

            elif e.kind == "slash":
                prog = min(1.0, e.progress)
                ex   = int(e.x1 + (e.x2 - e.x1) * prog)
                ey   = int(e.y1 + (e.y2 - e.y1) * prog)
                cv2.line(frame, (int(e.x1),int(e.y1)), (ex,ey), col,
                         max(1,int(e.width*e.life)), cv2.LINE_AA)
                cv2.line(frame, (int(e.x1),int(e.y1)), (ex,ey), (255,255,255), 1, cv2.LINE_AA)
                e.progress += e.speed

            elif e.kind == "rainbow_ring":
                r = int(e.r)
                steps = 36
                for i in range(steps):
                    hue = int((i / steps) * 180)
                    hsv_col = np.array([[[hue, 255, int(255*e.life)]]], dtype=np.uint8)
                    rgb_col = cv2.cvtColor(hsv_col, cv2.COLOR_HSV2BGR)[0][0]
                    a1 = (i / steps) * 2 * math.pi
                    a2 = ((i+1) / steps) * 2 * math.pi
                    p1 = (int(e.x + math.cos(a1)*r), int(e.y + math.sin(a1)*r))
                    p2 = (int(e.x + math.cos(a2)*r), int(e.y + math.sin(a2)*r))
                    cv2.line(frame, p1, p2, tuple(int(c) for c in rgb_col), e.width, cv2.LINE_AA)
                e.r += 4

            elif e.kind == "lightning":
                for (x1,y1,x2,y2) in e.branches:
                    cv2.line(frame, (int(x1),int(y1)), (int(x2),int(y2)),
                             col, max(1, int(2*e.life)), cv2.LINE_AA)
                    cv2.line(frame, (int(x1),int(y1)), (int(x2),int(y2)),
                             (255,255,255), 1, cv2.LINE_AA)

            e.life -= e.decay
            alive.append(e)
        self.effects = alive

    # ── HUD Drawing ───────────────────────────────────────────────

    def _draw_hud(self, frame, W, H, gesture, show_skeleton):
        font  = cv2.FONT_HERSHEY_SIMPLEX
        font2 = cv2.FONT_HERSHEY_DUPLEX

        # Top bar background
        cv2.rectangle(frame, (0,0), (W, 44), (10, 14, 26), -1)
        cv2.line(frame, (0, 44), (W, 44), (50,80,150), 1)

        # Title
        cv2.putText(frame, "GESTURE_MAGIC", (14, 30), font2, 0.7, (80,130,255), 1, cv2.LINE_AA)
        cv2.putText(frame, "v2.0 | MediaPipe", (220, 30), font, 0.4, (60,80,120), 1, cv2.LINE_AA)

        # FPS
        fps_color = (50,220,100) if self.fps >= 25 else (240,180,50)
        cv2.putText(frame, f"{self.fps} FPS", (W-90, 30), font2, 0.6, fps_color, 1, cv2.LINE_AA)

        # Skeleton toggle
        sk_label = "S:skeleton ON" if show_skeleton else "S:skeleton OFF"
        cv2.putText(frame, sk_label, (W-200, 30), font, 0.35, (60,80,120), 1, cv2.LINE_AA)

        # Gesture label (large, bottom center)
        if gesture and gesture in GESTURE_MAP and time.time() < self.gesture_label_until:
            color, label = GESTURE_MAP[gesture]
            text_size = cv2.getTextSize(label, font2, 1.0, 2)[0]
            tx = (W - text_size[0]) // 2
            ty = H - 30
            # Shadow
            cv2.putText(frame, label, (tx+2, ty+2), font2, 1.0, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(frame, label, (tx, ty), font2, 1.0, color, 2, cv2.LINE_AA)

        # Finger state bar (bottom-left) - primary hand
        names = ["T","I","M","R","P"]
        bx, by = 14, H - 14
        finger_state = self.finger_states[0] if self.finger_states[0] else [False]*5
        for i, (name, up) in enumerate(zip(names, finger_state)):
            col = (80,160,255) if up else (40,50,70)
            cv2.rectangle(frame, (bx + i*36, by-22), (bx + i*36+28, by), col, -1)
            cv2.rectangle(frame, (bx + i*36, by-22), (bx + i*36+28, by), (60,90,140), 1)
            cv2.putText(frame, name, (bx + i*36+8, by-7), font, 0.4, (255,255,255), 1, cv2.LINE_AA)

        # Bottom-right hint
        cv2.putText(frame, "Q: Quit | S: Skeleton", (W-220, H-10), font, 0.38, (50,70,110), 1, cv2.LINE_AA)

    def _show_label(self, gesture):
        self.gesture_label_until = time.time() + 2.0


# ══════════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = GestureMagic()
    app.run()
