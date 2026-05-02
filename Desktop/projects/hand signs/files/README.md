# 🖐 GestureMagic — Real-Time Hand Gesture Recognition

> Turn your hands into a magic wand. Detect gestures live from your webcam and trigger stunning animated visual effects — all in real time.

---

## ✨ Features

| Feature | Web Version | Python Version |
|---|---|---|
| MediaPipe Hands (21 landmarks) | ✅ | ✅ |
| Hand skeleton overlay | ✅ | ✅ |
| 10 gesture types | ✅ | ✅ |
| Animated FX per gesture | ✅ | ✅ |
| Swipe detection | ✅ | ✅ |
| Finger state indicators | ✅ | ✅ |
| FPS counter | ✅ | ✅ |
| Dark HUD / modern UI | ✅ | ✅ |
| Sound effects (Web Audio API) | ✅ | ❌ |
| Settings toggles | ✅ | Partial |
| No server required | ✅ | N/A |

---

## 📁 Project Structure

```
gesture-magic/
├── index.html          # 🌐 Complete browser app (no build step!)
├── gesture_magic.py    # 🐍 Python + OpenCV + MediaPipe version
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🚀 Quick Start

### Option A: Web Browser (Recommended — Zero Install)

1. Open `index.html` directly in **Chrome** or **Edge** (v90+)
2. Click **"ACTIVATE CAMERA"**
3. Allow camera access when prompted
4. Wave your hand in front of the webcam!

> **Note:** Must be opened via file:// or a local server. Some browsers restrict camera access on file:// — if that happens, run: `python3 -m http.server 8080` then visit `http://localhost:8080`

### Option B: Python + OpenCV

**Step 1: Install dependencies**

```bash
pip install opencv-python mediapipe numpy
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

**Step 2: Run the app**

```bash
python gesture_magic.py
```

**Step 3: Controls**

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Toggle skeleton overlay |

---

## 🤌 Recognized Gestures & Effects

| Gesture | How to Make It | Visual Effect |
|---------|---------------|---------------|
| **Open Palm** 🖐 | All 5 fingers extended | Radial energy wave burst (blue glow) |
| **Fist** ✊ | All fingers closed | Triple concentric shockwaves + sparks |
| **Thumbs Up** 👍 | Thumb up, others closed | Green aura particles rising |
| **Pointing** ☝️ | Only index finger up | Golden laser beam from fingertip |
| **Peace Sign** ✌️ | Index + middle up | Rainbow prismatic burst |
| **Pinch** 🤌 | Thumb + index touching | Cyan vortex spiraling inward |
| **OK Sign** 👌 | Thumb+index pinched, others open | 5 resonance rings |
| **Swipe Right** 👉 | Move hand quickly left→right | Orange slash trail |
| **Swipe Left** 👈 | Move hand quickly right→left | Purple slash trail |
| **Rock On** 🤘 | Index + pinky up | Yellow lightning bolt upward |

---

## 🏗 Technical Architecture

### Web Version Stack
```
Browser Camera API (getUserMedia)
    ↓
MediaPipe Hands WASM Model (CDN)
    ↓ 21 3D landmarks @ 30+ FPS
Gesture Classifier (pure JS)
    ↓ gesture name + confidence
Canvas 2D Renderer
  ├── Landmark Canvas  (skeleton overlay)
  └── FX Canvas        (particle system + effects)
```

### Python Version Stack
```
OpenCV VideoCapture (webcam)
    ↓
MediaPipe Hands Python API
    ↓ 21 3D landmarks
Gesture Classifier (Python)
    ↓
OpenCV drawing functions
  └── Overlaid particle/effect system
```

### Gesture Classification Logic

Each gesture is detected through a rule-based classifier operating on normalized (0–1) landmark coordinates:

```
Finger Extension:  tip.y < pip.y  (screen coords, y increases downward)
Thumb Extension:   tip.x > ip.x   (side comparison for mirrored frame)
Pinch Distance:    Euclidean(tip[4], tip[8]) < 0.06
Swipe:             ΔX over last 15 frames > 0.10 (10% of frame width)
```

### FX Particle System

Each effect is a combination of:
- **Particles**: position, velocity, gravity, decay, color — updated per-frame
- **Effects**: shockwaves (expanding rings), laser beams, slash trails, lightning bolt branches

---

## ⚡ Performance Tips

- **Lighting**: Ensure your hand is well-lit (avoid backlit setups)
- **Background**: Plain, dark backgrounds improve detection accuracy
- **Distance**: Keep hand 30–80 cm from camera
- **Browser**: Chrome/Edge outperform Firefox for WebAssembly
- **Model complexity**: Set to `0` (lite) for slower computers:
  - Web: change `modelComplexity: 1` → `0` in `index.html`
  - Python: change `model_complexity=1` → `0` in `gesture_magic.py`
- **Camera resolution**: Lower to 640×480 if you see lag

---

## 🔮 Future Upgrade Ideas

### 🎮 Gaming
- **Game controller**: Map gestures to keyboard events → control retro games
- **VR input**: Stream hand positions to WebXR for hand tracking in 3D
- **Rhythm game**: Tap gestures on beat to a music track

### 🎤 Presentation Control
- **Slide remote**: Swipe left/right to advance/go back in slides
- **Zoom/Pan**: Pinch gesture to zoom into sections
- **Laser pointer**: Pointing gesture controls a red dot on screen

### 👾 Avatar / Character Control
- **2D character**: Map hand poses to sprite animations (idle, run, attack)
- **Live2D avatar**: Drive a VTuber model with hand gestures
- **Puppet show**: Control multiple puppets with left/right hands

### 🤖 AI Upgrades
- **Gesture sequences**: Train a LSTM/Transformer on gesture streams for "combo" moves
- **Custom gestures**: Fine-tune MediaPipe with your own gesture dataset (MediaPipe Model Maker)
- **Two-hand support**: Enable `maxNumHands: 2` for dual-hand gestures (e.g., heart ❤️)
- **Pose fusion**: Combine hand + body pose for full-body gesture control

### 🎨 Creative
- **Air drawing**: Use pointing gesture to draw on a virtual canvas
- **Music theremin**: Map palm height → pitch, palm width → volume
- **Emoji generator**: Snap gesture → GIF auto-generated and copied to clipboard
- **Filter effects**: Map gestures to real-time video filters (pixelate, glitch, etc.)

### 📡 Integration
- **WebSocket server**: Broadcast gesture events to other devices on LAN
- **MIDI controller**: Map gestures to MIDI CC messages for music production
- **Smart home**: Trigger IoT automations via gesture → webhook
- **OBS plugin**: Use gesture events as scene/source triggers in streaming software

---

## 📦 requirements.txt

```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
```

---

## 🧠 MediaPipe Hand Landmarks Reference

```
        8   12  16  20
        |   |   |   |
        7   11  15  19
        |   |   |   |
        6   10  14  18
        |   |   |   |
    4   5   9   13  17
    |   |
    3   |
    |   |
    2   |
     \  |
      1  |
       \ |
        0 (WRIST)
```

Landmark indices: 0=Wrist, 1–4=Thumb, 5–8=Index, 9–12=Middle, 13–16=Ring, 17–20=Pinky

---

## 📜 License

MIT License — free to use, modify, and distribute.

Built with ❤️ using [MediaPipe](https://mediapipe.dev) + Canvas API / OpenCV
```
