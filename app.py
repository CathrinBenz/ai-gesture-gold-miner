import os
import urllib.request
import math
import random
import sys
import cv2
import pygame
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =====================================================================
# 1. PROCEDURAL SOUND SYNTHESIS (Your original sound engine)
# =====================================================================
pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.init()

def synthesize_tone(frequencies, durations, volumes=None):
    sample_rate = 44100
    all_chunks = []
    if volumes is None:
        volumes = [0.4] * len(frequencies)
    for freq, dur, vol in zip(frequencies, durations, volumes):
        n_samples = int(sample_rate * dur)
        t = np.linspace(0, dur, n_samples, False)
        envelope = np.exp(-3.2 * t / dur)
        wave = np.sin(2 * np.pi * freq * t) * envelope * vol
        all_chunks.append(wave)
    merged = np.concatenate(all_chunks)
    stereo = np.repeat(merged[:, np.newaxis], 2, axis=1)
    return pygame.sndarray.make_sound((stereo * 32767).astype(np.int16))

def synthesize_penalty():
    sample_rate = 44100
    duration = 0.35
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)
    buzz = (np.sin(2 * np.pi * 105 * t) + np.sin(2 * np.pi * 112 * t)) * 0.5
    stereo = np.repeat((buzz * np.exp(-2.0 * t / duration) * 0.45)[:, np.newaxis], 2, axis=1)
    return pygame.sndarray.make_sound((stereo * 32767).astype(np.int16))

def synthesize_explosion():
    sample_rate = 44100
    duration = 0.55
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)
    noise = np.random.uniform(-1, 1, n_samples)
    rumble = np.sin(2 * np.pi * 50 * t) * 0.7
    combined = (noise * 0.6 + rumble) * np.exp(-3.8 * t / duration) * 0.6
    stereo = np.repeat(combined[:, np.newaxis], 2, axis=1)
    return pygame.sndarray.make_sound((stereo * 32767).astype(np.int16))

def synthesize_launch():
    sample_rate = 44100
    duration = 0.2
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)
    freq = np.linspace(500, 160, n_samples)
    wave = np.sin(2 * np.pi * freq * t) * np.exp(-2.5 * t / duration) * 0.35
    stereo = np.repeat(wave[:, np.newaxis], 2, axis=1)
    return pygame.sndarray.make_sound((stereo * 32767).astype(np.int16))

SND_SHOOT = synthesize_launch()
SND_GOLD = synthesize_tone([880, 1174, 1567], [0.08, 0.08, 0.22], [0.4, 0.4, 0.5])
SND_DIAMOND = synthesize_tone([1318, 1760, 2349, 2793], [0.06, 0.06, 0.08, 0.3], [0.35, 0.4, 0.45, 0.5])
SND_PENALTY = synthesize_penalty()
SND_GRAB = synthesize_tone([170], [0.08], [0.5])
SND_BOOM = synthesize_explosion()
SND_WIN = synthesize_tone([523, 659, 784, 1046], [0.1, 0.1, 0.1, 0.4], [0.4, 0.4, 0.4, 0.5])
SND_LOSE = synthesize_tone([440, 415, 392, 349], [0.18, 0.18, 0.18, 0.5], [0.45, 0.45, 0.45, 0.5])

# =====================================================================
# 2. MEDIAPIPE TASKS SETUP
# =====================================================================
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task...")
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(MODEL_PATH, "wb") as f:
        f.write(resp.read())

detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        running_mode=vision.RunningMode.IMAGE
    )
)
cap = cv2.VideoCapture(0)

# =====================================================================
# 3. DISPLAY & PROCEDURAL SPRITES (Your exact graphics)
# =====================================================================
WIDTH, HEIGHT = 980, 700
BORDER_WIDTH = 14
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Gold Miner")
clock = pygame.time.Clock()

font = pygame.font.SysFont("Arial", 16, bold=True)
hud_font = pygame.font.SysFont("Arial", 20, bold=True)
big_font = pygame.font.SysFont("Arial", 28, bold=True)
title_font = pygame.font.SysFont("Impact", 34)
overlay_font = pygame.font.SysFont("Arial", 46, bold=True)

SKY_COLOR = (115, 185, 235)
GROUND_COLOR = (24, 16, 10)
BORDER_COLOR = (95, 58, 30)
BORDER_EDGE = (60, 32, 14)
ROPE_COLOR = (220, 205, 170)
WHITE = (255, 255, 255)

def make_gold_sprite(radius, seed_val=0):
    random.seed(seed_val)
    size = radius * 2 + 10
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    cx, cy = size // 2, size // 2
    num_pts = 9
    outer_pts = []
    for i in range(num_pts):
        angle = (2 * math.pi / num_pts) * i
        r = radius * random.uniform(0.85, 1.15)
        outer_pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))

    pygame.draw.polygon(surf, (180, 120, 10), outer_pts)
    pygame.draw.polygon(surf, (110, 70, 5), outer_pts, 2)
    mid_pts = [(cx + (px - cx) * 0.78, cy + (py - cy) * 0.78) for (px, py) in outer_pts]
    pygame.draw.polygon(surf, (255, 215, 0), mid_pts)
    high_pts = [
        (cx - radius * 0.5, cy - radius * 0.4),
        (cx + radius * 0.2, cy - radius * 0.6),
        (cx + radius * 0.4, cy - radius * 0.1),
        (cx - radius * 0.1, cy + radius * 0.1)
    ]
    pygame.draw.polygon(surf, (255, 245, 140), high_pts)
    pygame.draw.circle(surf, (255, 255, 255), (int(cx - radius * 0.3), int(cy - radius * 0.35)), max(2, radius // 7))
    return surf

def make_rock_sprite(radius, seed_val=0):
    random.seed(seed_val)
    size = radius * 2 + 12
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    cx, cy = size // 2, size // 2
    num_pts = 8
    outer_pts = []
    for i in range(num_pts):
        angle = (2 * math.pi / num_pts) * i
        r = radius * random.uniform(0.80, 1.20)
        outer_pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))

    pygame.draw.polygon(surf, (95, 95, 100), outer_pts)
    pygame.draw.polygon(surf, (50, 50, 55), outer_pts, 2)
    facet = [outer_pts[0], outer_pts[1], (cx, cy), outer_pts[7]]
    pygame.draw.polygon(surf, (135, 135, 142), facet)
    pygame.draw.line(surf, (45, 45, 48), (cx - radius * 0.4, cy - radius * 0.3), (cx + radius * 0.1, cy + radius * 0.4), 2)
    pygame.draw.line(surf, (45, 45, 48), (cx + radius * 0.1, cy + radius * 0.4), (cx + radius * 0.5, cy + radius * 0.2), 2)
    return surf

def make_diamond_sprite(radius):
    size = radius * 2 + 10
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    cx, cy = size // 2, size // 2
    pts = [(cx, cy - radius), (cx + radius, cy - radius * 0.3), (cx, cy + radius), (cx - radius, cy - radius * 0.3)]
    pygame.draw.polygon(surf, (60, 220, 245), pts)
    pygame.draw.polygon(surf, (190, 250, 255), [(cx, cy - radius), (cx + radius, cy - radius * 0.3), (cx, cy)])
    pygame.draw.polygon(surf, (240, 255, 255), [(cx, cy - radius), (cx, cy), (cx - radius, cy - radius * 0.3)])
    pygame.draw.polygon(surf, (255, 255, 255), pts, 2)
    return surf

def make_tnt_sprite(radius):
    size = radius * 2 + 8
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    rect = pygame.Rect(4, 4, size - 8, size - 8)
    pygame.draw.rect(surf, (215, 35, 30), rect, border_radius=4)
    pygame.draw.line(surf, (60, 60, 60), (4, size // 3), (size - 4, size // 3), 3)
    pygame.draw.line(surf, (60, 60, 60), (4, 2 * size // 3), (size - 4, 2 * size // 3), 3)
    pygame.draw.rect(surf, (40, 10, 10), rect, 2, border_radius=4)
    lbl = font.render("TNT", True, (255, 255, 255))
    surf.blit(lbl, lbl.get_rect(center=(size // 2, size // 2)))
    return surf

def make_skull_sprite(radius):
    size = radius * 2 + 10
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    cx, cy = size // 2, size // 2
    pygame.draw.circle(surf, (150, 75, 190), (cx, cy - 2), radius)
    pygame.draw.rect(surf, (150, 75, 190), (cx - radius * 0.6, cy + 2, radius * 1.2, radius * 0.7))
    pygame.draw.circle(surf, (20, 5, 30), (int(cx - radius * 0.38), cy - 1), max(2, radius // 4))
    pygame.draw.circle(surf, (20, 5, 30), (int(cx + radius * 0.38), cy - 1), max(2, radius // 4))
    lbl = font.render("-150", True, (255, 190, 220))
    surf.blit(lbl, lbl.get_rect(center=(cx, cy + radius * 0.4)))
    pygame.draw.circle(surf, (220, 140, 255), (cx, cy - 2), radius, 2)
    return surf

SPRITE_DIAMOND = make_diamond_sprite(12)
SPRITE_TNT = make_tnt_sprite(22)
SPRITE_SKULL = make_skull_sprite(18)

# =====================================================================
# 4. SPARKLES & PARTICLES
# =====================================================================
sparkles = []
particles = []
floating_texts = []

class Sparkle:
    def __init__(self, x, y, max_size=10, color=(255, 255, 255)):
        self.x = x
        self.y = y
        self.max_size = max_size
        self.color = color
        self.life = 0
        self.duration = 24

    def update(self):
        self.life += 1

    def draw(self, surface):
        if self.life >= self.duration:
            return
        progress = self.life / self.duration
        cur_size = math.sin(progress * math.pi) * self.max_size
        if cur_size < 1:
            return
        pts = [
            (self.x, self.y - cur_size * 1.5),
            (self.x + cur_size * 0.3, self.y - cur_size * 0.3),
            (self.x + cur_size * 1.5, self.y),
            (self.x + cur_size * 0.3, self.y + cur_size * 0.3),
            (self.x, self.y + cur_size * 1.5),
            (self.x - cur_size * 0.3, self.y + cur_size * 0.3),
            (self.x - cur_size * 1.5, self.y),
            (self.x - cur_size * 0.3, self.y - cur_size * 0.3)
        ]
        pygame.draw.polygon(surface, self.color, pts)
        pygame.draw.circle(surface, (255, 255, 255), (int(self.x), int(self.y)), max(1, int(cur_size * 0.4)))

class Particle:
    def __init__(self, x, y, color, vel_range=(-5, 5), decay=0.93, size=6):
        self.x = x
        self.y = y
        self.vx = random.uniform(vel_range[0], vel_range[1])
        self.vy = random.uniform(vel_range[0], vel_range[1])
        self.color = color
        self.size = size
        self.decay = decay

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.size *= self.decay

class FloatingText:
    def __init__(self, text, x, y, color):
        self.text = text
        self.x = x
        self.y = y
        self.color = color
        self.life = 45

    def update(self):
        self.y -= 1.3
        self.life -= 1

# =====================================================================
# 5. GAME OBJECTS & LEVEL STATE
# =====================================================================
GROUND_LEVEL = 105
PIVOT_X, PIVOT_Y = WIDTH // 2, GROUND_LEVEL
INIT_HOOK_LEN = 45
hook_len = INIT_HOOK_LEN
hook_angle = 0.0
swing_direction = 1
swing_speed = 0.03
max_swing_angle = math.radians(68)

claw_state = 'SWINGING'
shoot_speed = 16
retract_speed = 10
caught_item = None

# --- [BUTTONS & GAME STATES] ---
game_mode = 'START'  # Starts on the START screen first
btn_start = pygame.Rect(WIDTH // 2 - 130, HEIGHT // 2 - 35, 260, 70)  # Start button in center
btn_quit = pygame.Rect(20, 15, 100, 36)                                # Quit button in game
cursor_x, cursor_y = WIDTH // 2, HEIGHT // 2

current_level = 1
score = 0
target_score = 750
level_timer = 60
dynamite_count = 3
last_tick = pygame.time.get_ticks()
transition_cooldown = 0

class Mineral:
    def __init__(self, x, y, radius, m_type, value, weight, sprite):
        self.x = x
        self.y = y
        self.radius = radius
        self.m_type = m_type
        self.value = value
        self.weight = weight
        self.sprite = sprite

    def draw(self, surface):
        rect = self.sprite.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(self.sprite, rect.topleft)

class Pig:
    def __init__(self, y, speed, has_diamond=False):
        self.radius = 18
        self.has_diamond = has_diamond
        self.speed = speed
        self.x = random.randint(BORDER_WIDTH + 60, WIDTH - BORDER_WIDTH - 60)
        self.y = y
        self.m_type = "Diamond Pig" if has_diamond else "Wild Pig"
        self.value = 650 if has_diamond else -75
        self.weight = 0.9 if has_diamond else 1.7

    def update(self):
        self.x += self.speed
        if self.x <= BORDER_WIDTH + self.radius + 4:
            self.x = BORDER_WIDTH + self.radius + 4
            self.speed *= -1
        elif self.x >= WIDTH - BORDER_WIDTH - self.radius - 4:
            self.x = WIDTH - BORDER_WIDTH - self.radius - 4
            self.speed *= -1

    def draw(self, surface):
        pygame.draw.ellipse(surface, (255, 180, 190), (int(self.x - 20), int(self.y - 13), 40, 26))
        pygame.draw.ellipse(surface, (180, 110, 120), (int(self.x - 20), int(self.y - 13), 40, 26), 2)
        snout_offset = 14 if self.speed > 0 else -18
        pygame.draw.circle(surface, (255, 140, 160), (int(self.x + snout_offset), int(self.y + 2)), 6)
        eye_offset = 6 if self.speed > 0 else -10
        pygame.draw.circle(surface, (0, 0, 0), (int(self.x + eye_offset), int(self.y - 4)), 3)
        if self.has_diamond:
            rect = SPRITE_DIAMOND.get_rect(center=(int(self.x), int(self.y - 15)))
            surface.blit(SPRITE_DIAMOND, rect.topleft)

def setup_level(lvl):
    minerals = []
    for _ in range(1 + lvl):
        minerals.append(Mineral(
            random.randint(BORDER_WIDTH + 80, WIDTH - BORDER_WIDTH - 80),
            random.randint(GROUND_LEVEL + 90, HEIGHT - 100),
            radius=12, m_type="Diamond", value=600, weight=0.45, sprite=SPRITE_DIAMOND
        ))
    for i in range(3):
        sprite = make_gold_sprite(16, seed_val=i * 17 + lvl)
        minerals.append(Mineral(
            random.randint(BORDER_WIDTH + 80, WIDTH - BORDER_WIDTH - 80),
            random.randint(GROUND_LEVEL + 80, HEIGHT - 90),
            radius=16, m_type="Small Gold", value=100, weight=1.2, sprite=sprite
        ))
    for i in range(2 + (lvl // 2)):
        sprite = make_gold_sprite(30, seed_val=i * 29 + lvl)
        minerals.append(Mineral(
            random.randint(BORDER_WIDTH + 90, WIDTH - BORDER_WIDTH - 90),
            random.randint(GROUND_LEVEL + 130, HEIGHT - 90),
            radius=30, m_type="Large Gold", value=350, weight=3.8, sprite=sprite
        ))
    for i in range(2 + lvl):
        sprite = make_rock_sprite(24, seed_val=i * 37 + lvl)
        minerals.append(Mineral(
            random.randint(BORDER_WIDTH + 80, WIDTH - BORDER_WIDTH - 80),
            random.randint(GROUND_LEVEL + 90, HEIGHT - 90),
            radius=24, m_type="Rock", value=25, weight=4.5, sprite=sprite
        ))
    for _ in range(2):
        minerals.append(Mineral(
            random.randint(BORDER_WIDTH + 110, WIDTH - BORDER_WIDTH - 110),
            random.randint(GROUND_LEVEL + 110, HEIGHT - 110),
            radius=22, m_type="TNT", value=0, weight=1.0, sprite=SPRITE_TNT
        ))
    for _ in range(1 + lvl):
        minerals.append(Mineral(
            random.randint(BORDER_WIDTH + 90, WIDTH - BORDER_WIDTH - 90),
            random.randint(GROUND_LEVEL + 100, HEIGHT - 100),
            radius=18, m_type="Toxic Skull", value=-150, weight=2.0, sprite=SPRITE_SKULL
        ))

    pigs = [
        Pig(y=GROUND_LEVEL + 75, speed=2.2 + 0.3 * lvl, has_diamond=True),
        Pig(y=GROUND_LEVEL + 160, speed=-2.0 - 0.2 * lvl, has_diamond=False),
        Pig(y=HEIGHT - 120, speed=2.5 + 0.3 * lvl, has_diamond=False)
    ]
    return minerals, pigs

minerals, pigs = setup_level(current_level)

def trigger_explosion(center_x, center_y, radius=120):
    SND_BOOM.play()
    for _ in range(45):
        c = random.choice([(255, 60, 20), (255, 190, 0), (70, 70, 70)])
        particles.append(Particle(center_x, center_y, c, vel_range=(-7, 7), decay=0.91, size=random.randint(6, 12)))
    
    global minerals, pigs
    minerals = [m for m in minerals if math.hypot(m.x - center_x, m.y - center_y) >= radius]
    pigs = [p for p in pigs if math.hypot(p.x - center_x, p.y - center_y) >= radius]

# =====================================================================
# 6. GESTURE VISION RECOGNITION (Added Cursor Tracking)
# =====================================================================
def get_gestures_and_feed(frame):
    global cursor_x, cursor_y
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)

    pinch_trigger = False
    open_palm_trigger = False
    hand_seen = False
    h, w, _ = frame.shape

    if result.hand_landmarks:
        hand_seen = True
        for hand in result.hand_landmarks:
            wrist = hand[0]
            thumb = hand[4]
            index = hand[8]

            # Track hand coordinate for buttons
            cursor_x = int(cursor_x * 0.6 + (index.x * WIDTH) * 0.4)
            cursor_y = int(cursor_y * 0.6 + (index.y * HEIGHT) * 0.4)

            pinch_dist = math.hypot(thumb.x - index.x, thumb.y - index.y)
            if pinch_dist < 0.065:
                pinch_trigger = True

            fingers_extended = all(
                math.hypot(hand[tip].x - wrist.x, hand[tip].y - wrist.y) > 
                math.hypot(hand[tip - 2].x - wrist.x, hand[tip - 2].y - wrist.y) * 1.25
                for tip in [8, 12, 16, 20]
            )
            if fingers_extended and pinch_dist > 0.12:
                open_palm_trigger = True

            tx, ty = int(thumb.x * w), int(thumb.y * h)
            ix, iy = int(index.x * w), int(index.y * h)
            cv2.line(frame, (tx, ty), (ix, iy), (0, 255, 0) if pinch_trigger else (0, 160, 255), 2)
            cv2.circle(frame, (tx, ty), 5, (0, 255, 0), -1)
            cv2.circle(frame, (ix, iy), 5, (0, 255, 0), -1)

            if open_palm_trigger:
                cv2.putText(frame, "PALM: DYNAMITE", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 69, 255), 2)
            elif pinch_trigger:
                cv2.putText(frame, "PINCH: ACTION", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    pip_img = cv2.resize(frame, (210, 145))
    pip_img = cv2.cvtColor(pip_img, cv2.COLOR_BGR2RGB)
    pip_surface = pygame.surfarray.make_surface(pip_img.swapaxes(0, 1))

    return pip_surface, pinch_trigger, open_palm_trigger, hand_seen

# =====================================================================
# 7. MAIN ENGINE LOOP
# =====================================================================
running = True

try:
    while running:
        clock.tick(60)

        # -------------------------------------------------------------
        # EVENTS
        # -------------------------------------------------------------
        mouse_clicked = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_clicked = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if game_mode == 'START':
                        game_mode = 'PLAYING'
                    elif game_mode == 'PLAYING' and claw_state == 'SWINGING':
                        claw_state = 'SHOOTING'
                        SND_SHOOT.play()
                    elif game_mode in ['LEVEL_PASSED', 'GAME_OVER'] and transition_cooldown <= 0:
                        if game_mode == 'LEVEL_PASSED':
                            current_level += 1
                            target_score += 850
                        else:
                            current_level = 1
                            score = 0
                            target_score = 750
                        level_timer = 60
                        dynamite_count = 3
                        minerals, pigs = setup_level(current_level)
                        game_mode = 'PLAYING'
                elif event.key == pygame.K_d and game_mode == 'PLAYING' and claw_state == 'RETRACTING' and caught_item and dynamite_count > 0:
                    dynamite_count -= 1
                    trigger_explosion(caught_item.x, caught_item.y)
                    caught_item = None
                    retract_speed = 14

        # -------------------------------------------------------------
        # VISION INPUT
        # -------------------------------------------------------------
        ret, cam_frame = cap.read()
        pip_overlay = None
        pinch = False
        open_palm = False
        hand_seen = False

        if ret:
            pip_overlay, pinch, open_palm, hand_seen = get_gestures_and_feed(cam_frame)

        # -------------------------------------------------------------
        # STATE 1: START SCREEN (NEW START BUTTON)
        # -------------------------------------------------------------
        if game_mode == 'START':
            screen.fill(GROUND_COLOR)
            pygame.draw.rect(screen, SKY_COLOR, (0, 0, WIDTH, GROUND_LEVEL))
            pygame.draw.line(screen, (75, 45, 18), (0, GROUND_LEVEL), (WIDTH, GROUND_LEVEL), 4)

            # Plaque Title
            title_box = pygame.Rect(WIDTH // 2 - 180, 70, 360, 56)
            pygame.draw.rect(screen, (45, 25, 12), title_box, border_radius=8)
            pygame.draw.rect(screen, (218, 165, 32), title_box, 3, border_radius=8)
            gold_txt = title_font.render("AI GOLD MINER", True, (255, 225, 50))
            screen.blit(gold_txt, gold_txt.get_rect(center=title_box.center))

            # START BUTTON (Draw & Check Click)
            hovering_start = btn_start.collidepoint(cursor_x, cursor_y)
            btn_col = (45, 180, 60) if hovering_start else (30, 130, 40)
            pygame.draw.rect(screen, btn_col, btn_start, border_radius=12)
            pygame.draw.rect(screen, (255, 255, 255), btn_start, 3, border_radius=12)
            start_lbl = big_font.render("START GAME", True, WHITE)
            screen.blit(start_lbl, start_lbl.get_rect(center=btn_start.center))

            # Hint
            hint_txt = hud_font.render("Pinch hand over START to begin (or press SPACE)", True, (240, 240, 240))
            screen.blit(hint_txt, hint_txt.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 70)))

            if (hovering_start and pinch) or mouse_clicked:
                game_mode = 'PLAYING'
                transition_cooldown = 30

            # Hand Cursor
            if hand_seen:
                cur_col = (255, 50, 50) if pinch else (50, 255, 50)
                pygame.draw.circle(screen, cur_col, (cursor_x, cursor_y), 10, 2)
                pygame.draw.circle(screen, WHITE, (cursor_x, cursor_y), 3)

            if pip_overlay:
                pip_rect = pygame.Rect(WIDTH - 225, 12, 210, 145)
                screen.blit(pip_overlay, pip_rect.topleft)
                pygame.draw.rect(screen, WHITE, pip_rect, 2)

        # -------------------------------------------------------------
        # STATE 2: ACTIVE GAMEPLAY (WITH QUIT BUTTON)
        # -------------------------------------------------------------
        elif game_mode == 'PLAYING':
            # Check In-Game QUIT button click
            hovering_quit = btn_quit.collidepoint(cursor_x, cursor_y)
            if (hovering_quit and pinch and transition_cooldown <= 0) or (hovering_quit and mouse_clicked):
                running = False  # Quits the game

            if pygame.time.get_ticks() - last_tick >= 1000:
                level_timer -= 1
                last_tick = pygame.time.get_ticks()

            if pinch and claw_state == 'SWINGING' and not hovering_quit and transition_cooldown <= 0:
                claw_state = 'SHOOTING'
                SND_SHOOT.play()
            if open_palm and claw_state == 'RETRACTING' and caught_item and dynamite_count > 0:
                dynamite_count -= 1
                trigger_explosion(caught_item.x, caught_item.y)
                caught_item = None
                retract_speed = 14

            for p in pigs:
                if p != caught_item:
                    p.update()

            # Random Ambient Shining Glints on Valuables
            if random.random() < 0.28:
                valuable_candidates = [m for m in minerals if "Gold" in m.m_type or m.m_type == "Diamond"]
                if valuable_candidates:
                    chosen = random.choice(valuable_candidates)
                    rx = chosen.x + random.uniform(-chosen.radius * 0.7, chosen.radius * 0.7)
                    ry = chosen.y + random.uniform(-chosen.radius * 0.7, chosen.radius * 0.7)
                    sparkles.append(Sparkle(rx, ry, max_size=random.randint(7, 12),
                                            color=(255, 255, 200) if "Gold" in chosen.m_type else (200, 255, 255)))

            # Hook Mechanics
            if claw_state == 'SWINGING':
                hook_angle += swing_speed * swing_direction
                if abs(hook_angle) >= max_swing_angle:
                    swing_direction *= -1

            elif claw_state == 'SHOOTING':
                hook_len += shoot_speed
                hook_x = PIVOT_X + hook_len * math.sin(hook_angle)
                hook_y = PIVOT_Y + hook_len * math.cos(hook_angle)

                for m in minerals:
                    if math.hypot(hook_x - m.x, hook_y - m.y) < m.radius + 8:
                        minerals.remove(m)
                        if m.m_type == "TNT":
                            trigger_explosion(m.x, m.y)
                            claw_state = 'RETRACTING'
                            retract_speed = 12
                        else:
                            caught_item = m
                            claw_state = 'RETRACTING'
                            retract_speed = max(2.5, 14.0 / caught_item.weight)
                            SND_GRAB.play()
                        break

                if claw_state == 'SHOOTING':
                    for p in pigs:
                        if math.hypot(hook_x - p.x, hook_y - p.y) < p.radius + 10:
                            pigs.remove(p)
                            caught_item = p
                            claw_state = 'RETRACTING'
                            retract_speed = max(3.0, 14.0 / caught_item.weight)
                            SND_GRAB.play()
                            break

                if (hook_x <= BORDER_WIDTH + 6 or 
                    hook_x >= WIDTH - BORDER_WIDTH - 6 or 
                    hook_y >= HEIGHT - BORDER_WIDTH - 6):
                    claw_state = 'RETRACTING'
                    retract_speed = 11

            elif claw_state == 'RETRACTING':
                hook_len -= retract_speed
                if hook_len <= INIT_HOOK_LEN:
                    hook_len = INIT_HOOK_LEN
                    claw_state = 'SWINGING'

                    if caught_item:
                        score += caught_item.value
                        if caught_item.value >= 0:
                            if "Diamond" in caught_item.m_type:
                                SND_DIAMOND.play()
                                col = (80, 245, 255)
                            else:
                                SND_GOLD.play()
                                col = (255, 215, 0)
                            floating_texts.append(FloatingText(f"+${caught_item.value}", PIVOT_X - 20, PIVOT_Y + 15, col))
                            for _ in range(25):
                                particles.append(Particle(PIVOT_X, PIVOT_Y, col, vel_range=(-4, 4), decay=0.91))
                        else:
                            SND_PENALTY.play()
                            floating_texts.append(FloatingText(f"-${abs(caught_item.value)} PENALTY!", PIVOT_X - 45, PIVOT_Y + 15, (255, 60, 60)))
                            for _ in range(25):
                                particles.append(Particle(PIVOT_X, PIVOT_Y, (220, 40, 40), vel_range=(-3, 3), decay=0.92))
                        caught_item = None

            # Win/Lose Check
            remaining_valuables = sum(1 for m in minerals if "Gold" in m.m_type or m.m_type == "Diamond")
            remaining_valuables += sum(1 for p in pigs if p.has_diamond)

            if remaining_valuables == 0 or level_timer <= 0:
                if score >= target_score:
                    game_mode = 'LEVEL_PASSED'
                    SND_WIN.play()
                    transition_cooldown = 40
                else:
                    game_mode = 'GAME_OVER'
                    SND_LOSE.play()
                    transition_cooldown = 40

            # Rendering Pipeline
            screen.fill(GROUND_COLOR)
            pygame.draw.rect(screen, SKY_COLOR, (0, 0, WIDTH, GROUND_LEVEL))
            pygame.draw.line(screen, (75, 45, 18), (0, GROUND_LEVEL), (WIDTH, GROUND_LEVEL), 4)

            # Borders
            pygame.draw.rect(screen, BORDER_COLOR, (0, GROUND_LEVEL, BORDER_WIDTH, HEIGHT - GROUND_LEVEL))
            pygame.draw.rect(screen, BORDER_EDGE, (0, GROUND_LEVEL, BORDER_WIDTH, HEIGHT - GROUND_LEVEL), 2)
            pygame.draw.rect(screen, BORDER_COLOR, (WIDTH - BORDER_WIDTH, GROUND_LEVEL, BORDER_WIDTH, HEIGHT - GROUND_LEVEL))
            pygame.draw.rect(screen, BORDER_EDGE, (WIDTH - BORDER_WIDTH, GROUND_LEVEL, BORDER_WIDTH, HEIGHT - GROUND_LEVEL), 2)
            pygame.draw.rect(screen, BORDER_COLOR, (0, HEIGHT - BORDER_WIDTH, WIDTH, BORDER_WIDTH))
            pygame.draw.rect(screen, BORDER_EDGE, (0, HEIGHT - BORDER_WIDTH, WIDTH, BORDER_WIDTH), 2)

            for m in minerals:
                m.draw(screen)

            for p in pigs:
                p.draw(screen)

            # Hook & Winch
            hook_x = PIVOT_X + hook_len * math.sin(hook_angle)
            hook_y = PIVOT_Y + hook_len * math.cos(hook_angle)
            pygame.draw.line(screen, ROPE_COLOR, (PIVOT_X, PIVOT_Y), (hook_x, hook_y), 3)
            pygame.draw.circle(screen, (50, 50, 50), (PIVOT_X, PIVOT_Y), 16)
            pygame.draw.circle(screen, (220, 220, 220), (int(hook_x), int(hook_y)), 7)

            if caught_item:
                caught_item.x = hook_x
                caught_item.y = hook_y + caught_item.radius // 2
                caught_item.draw(screen)

            for s in sparkles[:]:
                s.update()
                s.draw(screen)
                if s.life >= s.duration:
                    sparkles.remove(s)

            for p in particles[:]:
                p.update()
                if p.size > 0.8:
                    pygame.draw.circle(screen, p.color, (int(p.x), int(p.y)), int(p.size))
                else:
                    particles.remove(p)

            for ft in floating_texts[:]:
                ft.update()
                if ft.life > 0:
                    screen.blit(hud_font.render(ft.text, True, ft.color), (ft.x, ft.y))
                else:
                    floating_texts.remove(ft)

            # DRAW IN-GAME QUIT BUTTON (Top Left)
            q_col = (200, 30, 30) if hovering_quit else (140, 20, 20)
            pygame.draw.rect(screen, q_col, btn_quit, border_radius=8)
            pygame.draw.rect(screen, WHITE, btn_quit, 2, border_radius=8)
            quit_lbl = hud_font.render("QUIT", True, WHITE)
            screen.blit(quit_lbl, quit_lbl.get_rect(center=btn_quit.center))

            # HUD Stats
            screen.blit(big_font.render(f"Level {current_level}", True, WHITE), (135, 12))
            score_col = (255, 230, 0) if score >= target_score else (235, 180, 70)
            screen.blit(hud_font.render(f"Score: ${score} / ${target_score}", True, score_col), (135, 46))
            timer_col = (255, 50, 50) if level_timer <= 10 else (30, 30, 30)
            screen.blit(font.render(f"Time: {level_timer}s  |  TNT: {dynamite_count}x [Palm/'D']", True, timer_col), (135, 74))

            # Top Center Title
            title_box = pygame.Rect(WIDTH // 2 - 145, 12, 290, 46)
            pygame.draw.rect(screen, (45, 25, 12), title_box, border_radius=8)
            pygame.draw.rect(screen, (218, 165, 32), title_box, 3, border_radius=8)
            screen.blit(big_font.render("AI GOLD MINER", True, (255, 225, 50)), (WIDTH // 2 - 95, 18))

            if random.random() < 0.12:
                glint_x = WIDTH // 2 + random.randint(-120, 120)
                sparkles.append(Sparkle(glint_x, 35, max_size=9, color=(255, 255, 220)))

            if pip_overlay:
                pip_rect = pygame.Rect(WIDTH - 225, 12, 210, 145)
                screen.blit(pip_overlay, pip_rect.topleft)
                pygame.draw.rect(screen, WHITE, pip_rect, 2)

            # Hand Cursor
            if hand_seen:
                cur_col = (255, 50, 50) if pinch else (50, 255, 50)
                pygame.draw.circle(screen, cur_col, (cursor_x, cursor_y), 10, 2)
                pygame.draw.circle(screen, WHITE, (cursor_x, cursor_y), 3)

        # -------------------------------------------------------------
        # STATE 3: LEVEL PASSED & GAME OVER
        # -------------------------------------------------------------
        elif game_mode in ['LEVEL_PASSED', 'GAME_OVER']:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 195))
            screen.blit(overlay, (0, 0))

            if game_mode == 'LEVEL_PASSED':
                t_pass = overlay_font.render("LEVEL PASSED!", True, (50, 255, 100))
                t_score = big_font.render(f"Total Revenue: ${score}  |  Goal Met: ${target_score}", True, WHITE)
                t_prompt = hud_font.render("Pinch Fingers or Press SPACE for Next Level", True, (255, 230, 50))
            else:
                t_pass = overlay_font.render("GAME OVER / FAILED", True, (255, 60, 60))
                t_score = big_font.render(f"Final Score: ${score}  |  Required Goal: ${target_score}", True, (220, 220, 220))
                t_prompt = hud_font.render("Pinch Fingers or Press SPACE to Restart", True, (255, 230, 50))

            screen.blit(t_pass, t_pass.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 60)))
            screen.blit(t_score, t_score.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 5)))
            screen.blit(t_prompt, t_prompt.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 65)))

            if pinch and transition_cooldown <= 0:
                if game_mode == 'LEVEL_PASSED':
                    current_level += 1
                    target_score += 850
                else:
                    current_level = 1
                    score = 0
                    target_score = 750
                level_timer = 60
                dynamite_count = 3
                minerals, pigs = setup_level(current_level)
                game_mode = 'PLAYING'
                transition_cooldown = 30

            if pip_overlay:
                pip_rect = pygame.Rect(WIDTH - 225, 12, 210, 145)
                screen.blit(pip_overlay, pip_rect.topleft)
                pygame.draw.rect(screen, WHITE, pip_rect, 2)

        if transition_cooldown > 0:
            transition_cooldown -= 1

        pygame.display.flip()

finally:
    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    sys.exit()
