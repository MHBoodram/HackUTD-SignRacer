import hashlib
import re
import sys
import threading
import time
import json
import os
import random

import cv2
import joblib
import pygame
import requests

from src.vision.pipeline import FEATURE_DIM, create_holistic, process_frame

# ============================ CONFIG ============================
# --- SpacetimeDB ---
STDB_URL      = "https://maincloud.spacetimedb.com"
DB_NAME       = "signracer-1exui"
POLL_INTERVAL = 0.25               # seconds between opponent updates
# Names must match `spacetime describe signracer-1exui --server maincloud`
# (case-sensitive; newer versions usually use snake_case)
REDUCER_CREATE   = "create_room"
REDUCER_JOIN     = "join_room"
REDUCER_PROGRESS = "set_progress"
TABLE_ROOM       = "room"
COLUMN_CODE      = "code"
HTTP_TIMEOUT  = 10

# --- Window ---
SCREEN_WIDTH  = 1280
SCREEN_HEIGHT = 720
FPS           = 60
WINDOW_CAPTION = "SignRacer"

# --- Title ---
TITLE_TEXT       = "SignRacer"
TITLE_FONT_SIZE  = 96
TITLE_COLOR      = (255, 255, 255)
TITLE_Y          = 90

# --- Colors ---
BG_COLOR          = (24, 26, 36)
BUTTON_COLOR      = (70, 90, 200)
BUTTON_HOVER      = (100, 120, 235)
BUTTON_DISABLED   = (60, 62, 80)
BUTTON_TEXT_COLOR = (255, 255, 255)
UNTYPED_COLOR     = (120, 125, 145)
BOTH_DONE_COLOR   = (200, 200, 200)
P1_COLOR          = (80, 200, 120)    # you
P2_COLOR          = (240, 120, 90)    # opponent
ERROR_COLOR       = (240, 90, 90)

# --- Menu buttons ---
BUTTON_WIDTH     = 360
BUTTON_HEIGHT    = 80
BUTTON_SPACING   = 30
BUTTON_FONT_SIZE = 36
BUTTON_START_Y   = 200
CREATE_BUTTON_TEXT = "Create Room"
JOIN_BUTTON_TEXT   = "Join with Code"

# --- Room code ---
ROOM_CODE_LENGTH = 5
ROOM_CODE_CHARS  = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CODE_FONT_SIZE   = 48

# --- Webcam (top right) ---
WEBCAM_INDEX     = 0
WEBCAM_SIZE      = (360, 270)
WEBCAM_MARGIN    = 20
WEBCAM_MIRROR    = True
WEBCAM_BORDER    = 3
WEBCAM_BORDER_COLOR = (255, 255, 255)

# --- Typing text (bottom) ---
TEXT_FONT_SIZE    = 40
TEXT_LINE_SPACING = 64
TEXT_AREA         = pygame.Rect(40, 380, SCREEN_WIDTH - 80, 300)
# Race text: one phrase is picked per room (both players get the same one).
# Add your own: lowercase letters and spaces only.
GAME_PHRASES = [
    "the quick brown fox jumps over the lazy dog",
    "sign language is a beautiful way to talk",
    "practice makes perfect",
    "hands up for a fast race",
    "never stop learning something new",
    "keep your fingers steady and clear",
    "a journey of a thousand miles begins with one step",
    "speed and accuracy win the race",
    "hello how are you today",
    "good luck and have fun",
    "the early bird catches the worm",
    "every expert was once a beginner",
    "fingers moving fast and sure",
    "pack my box with five dozen liquor jugs",
    "how vexingly quick daft zebras jump",
    "sphinx of black quartz judge my vow",
    "we love hackathons and late night coding",
    "a picture is worth a thousand words",
    "actions speak louder than words",
    "time flies when you are having fun",
    "the sun rises in the east",
    "coffee and code keep us going",
    "clear hands make clear signs",
    "learn to sign and make a new friend",
    "the best way to learn is to practice",
    "smile and sign with confidence",
    "stay calm and keep signing",
    "the finish line is just ahead",
    "winners never quit and quitters never win",
    "one small step for a signer",
    "jump high run fast sign well",
    "patience and practice build skill",
    "slow is smooth and smooth is fast",
    "look at the camera and show your hand",
    "mountains are climbed one step at a time",
    "a quiet mind makes for steady hands",
    "ready set sign",
    "good things come to those who wait",
    "if at first you do not succeed try again",
    "knowledge is power",
]
CARET_THICKNESS   = 4

# --- Player labels / progress bars ---
PLAYER_NAMES     = ["You", "Opponent"]
PLAYER_COLORS    = [P1_COLOR, P2_COLOR]
LABEL_FONT_SIZE  = 28
BAR_POS          = (40, 40)
BAR_SIZE         = (500, 24)
BAR_SPACING      = 60
BAR_BG_COLOR     = (50, 54, 70)

# --- Game flow ---
COUNTDOWN_SECONDS = 3
COUNTDOWN_FONT_SIZE = 200
RESULT_FONT_SIZE    = 90

# --- Hints ---
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.join(BASE_DIR, "data", "asl_dataset")   # asl_dataset/<letter or number>/<HINT_FILENAME>
HINT_EXTENSIONS   = (".jpg", ".jpeg", ".png")
HINT_FILENAME     = "hand1_0_bot_seg_1_cropped.jpeg"   # the exact image to use in each letter folder
HINT_RANDOM_PICK  = False          # only used if HINT_FILENAME isn't found: False = first image, True = random
HINT_DEFAULT_ON   = True           # used until the player changes the checkbox
SETTINGS_FILE     = os.path.join(BASE_DIR, "signracer_settings.json")
HINT_SIZE         = (240, 240)     # hint box on screen (image is scaled to fit)
HINT_GAP          = 20             # gap between the hint box and the webcam
HINT_BOX_COLOR    = (50, 54, 70)
HINT_CHECKBOX_TEXT = "Show hints"
CHECKBOX_SIZE     = 28
CHECKBOX_COLOR    = (255, 255, 255)
CHECKBOX_CHECK_COLOR = (80, 200, 120)
CHECKBOX_GAME_POS = (40, 190)      # top-left of the checkbox during a game
CHECKBOX_MENU_Y   = 530            # y of the checkbox on the main menu
CHECKBOX_LOBBY_Y  = 570            # y of the checkbox in the lobby

# --- Practice mode ---
PRACTICE_BUTTON_TEXT = "Practice"
PRACTICE_MODES   = [("Letters", "abcdefghijklmnopqrstuvwxyz"),
                    ("Numbers", "0123456789"),
                    ("All",     "abcdefghijklmnopqrstuvwxyz0123456789")]   # Tab cycles through these
PRACTICE_PAUSE   = 0.7             # seconds to pause after a correct answer / skip
PRACTICE_TARGET_FONT_SIZE = 260
PRACTICE_TARGET_Y = 520            # vertical center of the big target letter
PRACTICE_STATS_POS = (40, 20)      # top-left of the stats text
PRACTICE_STATS_SPACING = 30

# --- Camera switching ---
CAMERA_BUTTON_TEXT = "Change camera"
CAMERA_BUTTON_SIZE = (230, 36)
CAMERA_BUTTON_GAP  = 8             # gap between the webcam and the button
CAMERA_MAX_INDEX   = 5             # highest camera index to try (0..5)

# --- Testing ---
DEBUG_KEYBOARD   = False           # True = type on keyboard to advance (for testing)
# ================================================================
# Run a second copy for testing on one PC:  python ui.py 2
# (each profile gets its own SpacetimeDB identity)
PROFILE = sys.argv[1] if len(sys.argv) > 1 else "1"


# ------------------------------------------------------------------
# SpacetimeDB networking (all HTTP runs in background threads)
# ------------------------------------------------------------------
# Room table column order (matches the C# struct):
#   0 Code, 1 P1, 2 P2, 3 P1Progress, 4 P2Progress
COL_CODE, COL_P1, COL_P2, COL_P1_PROG, COL_P2_PROG = range(5)


def _ident(cell):
    """Normalise an Identity cell from the SQL JSON into a lowercase hex string."""
    while isinstance(cell, (list, tuple)) and cell:
        cell = cell[0]
    if isinstance(cell, dict) and cell:
        cell = next(iter(cell.values()))
        return _ident(cell)
    return str(cell).lower().replace("0x", "")


def _option_is_none(cell):
    """SpacetimeDB encodes Option as [tag, value]; tag 1 = None."""
    return isinstance(cell, (list, tuple)) and len(cell) == 2 and cell[0] == 1


class Net:
    def __init__(self, profile):
        self.token_file = f"stdb_token_maincloud_{profile}.json"
        self.identity = None
        self.headers = None
        self.ready = False
        self.status = "Connecting..."      # shown in the UI
        self.error = ""                    # last user-facing error
        self.busy = False

        self.result = None                 # None / "created" / "joined"
        self.room_code = ""
        self.my_side = 0                   # 0 = created room (P1), 1 = joined (P2)
        self.opp_joined = False
        self.opp_progress = 0
        self.my_progress = 0

        self._run_id = 0
        threading.Thread(target=self._connect, daemon=True).start()

    # ---------- low-level ----------
    def _connect(self):
        try:
            if os.path.exists(self.token_file):
                data = json.load(open(self.token_file))
            else:
                r = requests.post(f"{STDB_URL}/v1/identity", timeout=HTTP_TIMEOUT)
                r.raise_for_status()
                data = r.json()
                json.dump(data, open(self.token_file, "w"))
            self.identity = data["identity"].lower().replace("0x", "")
            self.headers = {"Authorization": f"Bearer {data['token']}"}
            self.ready = True
            self.status = ""
        except Exception as e:
            self.status = f"Can't reach server: {e}"

    def _call(self, reducer, *args):
        r = requests.post(f"{STDB_URL}/v1/database/{DB_NAME}/call/{reducer}",
                          json=list(args), headers=self.headers, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            text = r.text.strip().splitlines()
            raise RuntimeError(text[0] if text else f"HTTP {r.status_code}")

    def _query(self, sql):
        r = requests.post(f"{STDB_URL}/v1/database/{DB_NAME}/sql",
                          data=sql, headers=self.headers, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code}: {r.text.strip()[:150]}")
        return r.json()[0]["rows"]

    # ---------- actions from the UI ----------
    def create_room(self):
        self._start(self._create_room)

    def join_room(self, code):
        self._start(self._join_room, code)

    def _start(self, fn, *args):
        if self.busy or not self.ready:
            return
        self.busy = True
        self.error = ""
        self.result = None
        threading.Thread(target=self._guard, args=(fn, *args), daemon=True).start()

    def _guard(self, fn, *args):
        try:
            fn(*args)
        except Exception as e:
            self.error = str(e)
        finally:
            self.busy = False

    def _create_room(self):
        before = {row[COL_CODE] for row in self._query(f"SELECT * FROM {TABLE_ROOM}")}
        self._call(REDUCER_CREATE)
        mine = [row[COL_CODE] for row in self._query(f"SELECT * FROM {TABLE_ROOM}")
                if _ident(row[COL_P1]) == self.identity]
        new = [c for c in mine if c not in before] or mine
        if not new:
            raise RuntimeError("Room created but code not found")
        self._enter_room(new[-1], side=0)
        self.result = "created"

    def _join_room(self, code):
        self._call(REDUCER_JOIN, code)
        self._enter_room(code, side=1)
        self.result = "joined"

    # ---------- room polling ----------
    def _enter_room(self, code, side):
        self.room_code = code
        self.my_side = side
        self.opp_joined = (side == 1)
        self.opp_progress = 0
        self.my_progress = 0
        self._run_id += 1
        threading.Thread(target=self._poll_loop, args=(self._run_id,), daemon=True).start()

    def leave(self):
        self._run_id += 1                  # stops the poll thread
        self.result = None
        self.room_code = ""

    def _poll_loop(self, run_id):
        last_sent = 0
        code = re.sub(r"[^A-Za-z0-9]", "", self.room_code)
        while run_id == self._run_id:
            try:
                if self.my_progress != last_sent:
                    sending = self.my_progress
                    self._call(REDUCER_PROGRESS, code, sending)
                    last_sent = sending
                rows = self._query(f"SELECT * FROM {TABLE_ROOM} WHERE {COLUMN_CODE} = '{code}'")
                if rows:
                    row = rows[0]
                    if self.my_side == 0:
                        self.opp_joined = not _option_is_none(row[COL_P2])
                        self.opp_progress = row[COL_P2_PROG]
                    else:
                        self.opp_progress = row[COL_P1_PROG]
            except Exception as e:
                self.status = f"Connection problem: {e}"
            time.sleep(POLL_INTERVAL)


# ------------------------------------------------------------------
# UI helpers
# ------------------------------------------------------------------
class Button:
    def __init__(self, text, center, font):
        self.text = text
        self.font = font
        self.rect = pygame.Rect(0, 0, BUTTON_WIDTH, BUTTON_HEIGHT)
        self.rect.center = center

    def draw(self, surf, enabled=True):
        hovered = enabled and self.rect.collidepoint(pygame.mouse.get_pos())
        color = BUTTON_DISABLED if not enabled else (BUTTON_HOVER if hovered else BUTTON_COLOR)
        pygame.draw.rect(surf, color, self.rect, border_radius=16)
        label = self.font.render(self.text, True, BUTTON_TEXT_COLOR)
        surf.blit(label, label.get_rect(center=self.rect.center))

    def clicked(self, event):
        return (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.rect.collidepoint(event.pos))


def pick_game_text(room_code):
    """Both players derive the same phrase from the shared room code."""
    digest = hashlib.sha256(room_code.encode("utf-8")).hexdigest()
    return GAME_PHRASES[int(digest, 16) % len(GAME_PHRASES)]


def layout_text(text, font, max_width):
    """Word-wrap and return [(char, x, y)] relative to the text area."""
    positions, x, y = [], 0, 0
    for word in text.split(" "):
        word_w = sum(font.size(c)[0] for c in word)
        space_w = font.size(" ")[0]
        if x + word_w > max_width and x > 0:
            x, y = 0, y + TEXT_LINE_SPACING
        for c in word:
            positions.append((c, x, y))
            x += font.size(c)[0]
        positions.append((" ", x, y))
        x += space_w
    return positions[:-1]


def frame_to_surface(frame):
    frame = cv2.resize(frame, WEBCAM_SIZE)
    if WEBCAM_MIRROR:
        frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(frame.tobytes(), WEBCAM_SIZE, "RGB")


def recognize_sign(frame, holistic, model):
    """Predict a sign from one webcam frame, or return None when no hand is detected."""
    features = process_frame(holistic, frame)
    if features is None:
        return None
    return str(model.predict(features.reshape(1, -1))[0])


def load_settings():
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(**updates):
    data = load_settings()
    data.update(updates)
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError:
        pass


def load_hint_setting():
    return bool(load_settings().get("hints", HINT_DEFAULT_ON))


def save_hint_setting(value):
    save_settings(hints=value)


class HintManager:
    """Finds one image per character in DATA_DIR/<char>/ and caches it."""

    def __init__(self):
        self.folders = {}          # lowercase char -> folder path
        self.cache = {}            # lowercase char -> Surface or None
        if os.path.isdir(DATA_DIR):
            for name in os.listdir(DATA_DIR):
                path = os.path.join(DATA_DIR, name)
                if os.path.isdir(path):
                    self.folders[name.lower()] = path

    def get(self, ch):
        ch = ch.lower()
        if ch not in self.cache:
            self.cache[ch] = self._load(ch)
        return self.cache[ch]

    def _load(self, ch):
        folder = self.folders.get(ch)
        if not folder:
            return None
        files = sorted(f for f in os.listdir(folder)
                       if f.lower().endswith(HINT_EXTENSIONS))
        if not files:
            return None
        exact = [f for f in files if f.lower() == HINT_FILENAME.lower()]
        if exact:
            pick = exact[0]
        else:                      # fallback if that file is missing in this folder
            pick = random.choice(files) if HINT_RANDOM_PICK else files[0]
        try:
            img = pygame.image.load(os.path.join(folder, pick)).convert()
        except pygame.error:
            return None
        w, h = img.get_size()
        scale = min(HINT_SIZE[0] / w, HINT_SIZE[1] / h)
        return pygame.transform.smoothscale(
            img, (max(1, int(w * scale)), max(1, int(h * scale))))


def draw_checkbox(screen, font, text, checked, topleft):
    box = pygame.Rect(topleft, (CHECKBOX_SIZE, CHECKBOX_SIZE))
    pygame.draw.rect(screen, BAR_BG_COLOR, box, border_radius=6)
    pygame.draw.rect(screen, CHECKBOX_COLOR, box, 2, border_radius=6)
    if checked:
        pts = [(box.x + 6, box.centery), (box.x + 12, box.bottom - 8),
               (box.right - 6, box.y + 7)]
        pygame.draw.lines(screen, CHECKBOX_CHECK_COLOR, False, pts, 4)
    label = font.render(text, True, TITLE_COLOR)
    screen.blit(label, label.get_rect(midleft=(box.right + 10, box.centery)))


class Practice:
    """State for the offline practice range (random letters/numbers)."""

    def __init__(self):
        self.mode = 0
        self.reset()

    def reset(self):
        self.score = 0
        self.streak = 0
        self.best = 0
        self.total_time = 0.0
        self.target = ""
        self.target_start = time.time()
        self.last_detected = ""
        self.flash_text = ""
        self.flash_color = P1_COLOR
        self.flash_until = 0.0
        self.next_target()

    def mode_name(self):
        return PRACTICE_MODES[self.mode][0]

    def next_target(self, pause=0.0):
        pool = PRACTICE_MODES[self.mode][1]
        choices = [c for c in pool if c != self.target] or list(pool)
        self.target = random.choice(choices)
        self.target_start = time.time() + pause

    def flash(self, text, color):
        self.flash_text = text
        self.flash_color = color
        self.flash_until = time.time() + PRACTICE_PAUSE

    def correct(self):
        elapsed = max(0.0, time.time() - self.target_start)
        self.score += 1
        self.streak += 1
        self.best = max(self.best, self.streak)
        self.total_time += elapsed
        self.flash(f"Correct!  {elapsed:.1f}s", P1_COLOR)
        self.next_target(PRACTICE_PAUSE)

    def skip(self):
        self.streak = 0
        self.flash(f"Skipped {self.target.upper()}", UNTYPED_COLOR)
        self.next_target(PRACTICE_PAUSE)

    def cycle_mode(self):
        self.mode = (self.mode + 1) % len(PRACTICE_MODES)
        self.target = ""
        self.next_target()


def draw_hint_box(screen, hints, ch, cam_rect, font):
    hint_rect = pygame.Rect(cam_rect.x - HINT_GAP - HINT_SIZE[0], cam_rect.y, *HINT_SIZE)
    pygame.draw.rect(screen, HINT_BOX_COLOR, hint_rect, border_radius=12)
    img = hints.get(ch)
    if img:
        screen.blit(img, img.get_rect(center=hint_rect.center))
    else:
        msg = font.render("No hint image", True, UNTYPED_COLOR)
        screen.blit(msg, msg.get_rect(center=hint_rect.center))
    tag = font.render(f"Sign: {ch.upper()}", True, P1_COLOR)
    screen.blit(tag, tag.get_rect(midtop=(hint_rect.centerx, hint_rect.bottom + 6)))


def draw_small_button(screen, font, text, rect, enabled=True):
    hovered = enabled and rect.collidepoint(pygame.mouse.get_pos())
    color = BUTTON_DISABLED if not enabled else (BUTTON_HOVER if hovered else BUTTON_COLOR)
    pygame.draw.rect(screen, color, rect, border_radius=10)
    label = font.render(text, True, BUTTON_TEXT_COLOR)
    screen.blit(label, label.get_rect(center=rect.center))


def draw_centered(screen, font, text, color, y):
    surf = font.render(text, True, color)
    screen.blit(surf, surf.get_rect(center=(SCREEN_WIDTH // 2, y)))


# ------------------------------------------------------------------
def main():
    model_path = os.path.join(BASE_DIR, "models", "asl_classifier.plk")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Generate the dataset and run `python -m src.vision.model_trainer` first."
        )
    model = joblib.load(model_path)
    if model.n_features_in_ != FEATURE_DIM:
        raise ValueError(
            f"Model expects {model.n_features_in_} features, but the vision pipeline "
            f"produces {FEATURE_DIM}. Regenerate the dataset and retrain the model."
        )
    holistic = create_holistic()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"{WINDOW_CAPTION} (profile {PROFILE})")
    clock = pygame.time.Clock()

    title_font  = pygame.font.Font(None, TITLE_FONT_SIZE)
    button_font = pygame.font.Font(None, BUTTON_FONT_SIZE)
    code_font   = pygame.font.Font(None, CODE_FONT_SIZE)
    label_font  = pygame.font.Font(None, LABEL_FONT_SIZE)
    big_font    = pygame.font.Font(None, COUNTDOWN_FONT_SIZE)
    result_font = pygame.font.Font(None, RESULT_FONT_SIZE)
    text_font   = pygame.font.SysFont("couriernew", TEXT_FONT_SIZE, bold=True)

    hints = HintManager()
    show_hints = load_hint_setting()   # stays on/off across rounds and restarts
    practice = Practice()
    cam_index = int(load_settings().get("camera", WEBCAM_INDEX))   # remembered between runs
    cam_state = {"switching": False, "pending": None, "msg": "", "msg_until": 0.0}
    target_font = pygame.font.Font(None, PRACTICE_TARGET_FONT_SIZE)

    cx = SCREEN_WIDTH // 2
    create_btn = Button(CREATE_BUTTON_TEXT, (cx, BUTTON_START_Y + BUTTON_HEIGHT // 2), button_font)
    join_btn = Button(JOIN_BUTTON_TEXT,
                      (cx, BUTTON_START_Y + BUTTON_HEIGHT * 3 // 2 + BUTTON_SPACING), button_font)
    practice_btn = Button(PRACTICE_BUTTON_TEXT,
                          (cx, BUTTON_START_Y + BUTTON_HEIGHT * 5 // 2 + BUTTON_SPACING * 2), button_font)

    net = Net(PROFILE)
    cap = None
    game_text = GAME_PHRASES[0]        # replaced by a room-based pick when a race starts
    char_layout = layout_text(game_text, text_font, TEXT_AREA.width)
    glyph_cache = {}

    def glyph(ch, color):
        key = (ch, color)
        if key not in glyph_cache:
            glyph_cache[key] = text_font.render(ch, True, color)
        return glyph_cache[key]

    state = "MENU"             # MENU, JOIN, LOBBY, GAME
    typed_code = ""
    progress = [0, 0]          # [you, opponent]
    webcam_surface = None
    game_start = 0.0
    winner = None              # None / "You" / "Opponent"

    def checkbox_rect():
        w = CHECKBOX_SIZE + 10 + label_font.size(HINT_CHECKBOX_TEXT)[0]
        if state == "MENU":
            return pygame.Rect(cx - w // 2, CHECKBOX_MENU_Y, w, CHECKBOX_SIZE)
        if state == "LOBBY":
            return pygame.Rect(cx - w // 2, CHECKBOX_LOBBY_Y, w, CHECKBOX_SIZE)
        if state in ("GAME", "PRACTICE"):
            return pygame.Rect(*CHECKBOX_GAME_POS, w, CHECKBOX_SIZE)
        return None

    def camera_button_rect():
        if state not in ("GAME", "PRACTICE"):
            return None
        r = pygame.Rect(0, 0, *CAMERA_BUTTON_SIZE)
        r.topright = (SCREEN_WIDTH - WEBCAM_MARGIN,
                      WEBCAM_MARGIN + WEBCAM_SIZE[1] + CAMERA_BUTTON_GAP)
        return r

    def switch_camera():
        """Look for the next working camera in a background thread."""
        if cam_state["switching"]:
            return
        cam_state["switching"] = True
        start = cam_index

        def worker():
            found = None
            for step in range(1, CAMERA_MAX_INDEX + 1):
                idx = (start + step) % (CAMERA_MAX_INDEX + 1)
                test = cv2.VideoCapture(idx)
                if test.isOpened() and test.read()[0]:
                    found = (test, idx)
                    break
                test.release()
            if found:
                save_settings(camera=found[1])
                cam_state["pending"] = found
            else:
                cam_state["msg"] = "No other camera found"
                cam_state["msg_until"] = time.time() + 3
            cam_state["switching"] = False

        threading.Thread(target=worker, daemon=True).start()

    def start_practice():
        nonlocal state, cap
        practice.reset()
        if cap is None:
            cap = cv2.VideoCapture(cam_index)
        state = "PRACTICE"

    def start_countdown():
        nonlocal state, game_start, winner, progress, cap, game_text, char_layout
        game_text = pick_game_text(net.room_code)
        char_layout = layout_text(game_text, text_font, TEXT_AREA.width)
        progress = [0, 0]
        net.my_progress = 0
        winner = None
        game_start = time.time() + COUNTDOWN_SECONDS
        if cap is None:
            cap = cv2.VideoCapture(cam_index)
        state = "GAME"

    def back_to_menu():
        nonlocal state
        net.leave()
        net.error = ""
        state = "MENU"

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                net.leave()
                if cap:
                    cap.release()
                holistic.close()
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                cb = checkbox_rect()
                if cb and cb.collidepoint(event.pos):
                    show_hints = not show_hints
                    save_hint_setting(show_hints)
                cam_btn = camera_button_rect()
                if cam_btn and cam_btn.collidepoint(event.pos):
                    switch_camera()

            if state == "MENU":
                if net.ready and not net.busy:
                    if create_btn.clicked(event):
                        net.create_room()
                    elif join_btn.clicked(event):
                        typed_code = ""
                        net.error = ""
                        state = "JOIN"
                if not net.busy and practice_btn.clicked(event):
                    start_practice()

            elif state == "JOIN":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        net.error = ""
                        state = "MENU"
                    elif event.key == pygame.K_BACKSPACE:
                        typed_code = typed_code[:-1]
                    elif (event.key == pygame.K_RETURN
                          and len(typed_code) == ROOM_CODE_LENGTH and not net.busy):
                        net.join_room(typed_code)
                    elif (event.unicode and event.unicode.upper() in ROOM_CODE_CHARS
                          and len(typed_code) < ROOM_CODE_LENGTH):
                        typed_code += event.unicode.upper()

            elif state in ("LOBBY", "GAME"):
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        back_to_menu()
                    elif (state == "GAME" and DEBUG_KEYBOARD and winner is None
                          and time.time() >= game_start
                          and progress[0] < len(game_text)
                          and event.unicode == game_text[progress[0]]):
                        progress[0] += 1

            elif state == "PRACTICE":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        state = "MENU"
                    elif event.key == pygame.K_RETURN:
                        practice.skip()
                    elif event.key == pygame.K_TAB:
                        practice.cycle_mode()
                    elif (DEBUG_KEYBOARD and event.unicode
                          and time.time() >= practice.flash_until
                          and event.unicode.lower() == practice.target):
                        practice.correct()

        # ---- react to network results ----
        if state in ("MENU", "JOIN") and net.result == "created":
            net.result = None
            state = "LOBBY"
        elif state in ("MENU", "JOIN") and net.result == "joined":
            net.result = None
            start_countdown()
        elif state == "LOBBY" and net.opp_joined:
            start_countdown()

        if cam_state["pending"]:
            new_cap, new_idx = cam_state["pending"]
            cam_state["pending"] = None
            if cap is not None:
                cap.release()
            cap, cam_index = new_cap, new_idx
            webcam_surface = None

        screen.fill(BG_COLOR)

        # ---------------- MENU ----------------
        if state == "MENU":
            draw_centered(screen, title_font, TITLE_TEXT, TITLE_COLOR, TITLE_Y)
            enabled = net.ready and not net.busy
            create_btn.draw(screen, enabled)
            join_btn.draw(screen, enabled)
            practice_btn.draw(screen, not net.busy)
            msg = net.error or net.status
            if net.busy:
                msg, col = "Working...", UNTYPED_COLOR
            else:
                col = ERROR_COLOR if net.error or "problem" in msg or "reach" in msg else UNTYPED_COLOR
            if msg:
                draw_centered(screen, label_font, msg, col, 590)

        # ---------------- JOIN ----------------
        elif state == "JOIN":
            draw_centered(screen, title_font, TITLE_TEXT, TITLE_COLOR, TITLE_Y)
            draw_centered(screen, button_font, "Enter room code, then press Enter", UNTYPED_COLOR, 240)
            box = pygame.Rect(0, 0, 360, 80)
            box.center = (cx, 330)
            pygame.draw.rect(screen, BAR_BG_COLOR, box, border_radius=12)
            code = code_font.render(typed_code, True, TITLE_COLOR)
            screen.blit(code, code.get_rect(center=box.center))
            if net.busy:
                draw_centered(screen, label_font, "Joining...", UNTYPED_COLOR, 410)
            elif net.error:
                draw_centered(screen, label_font, net.error, ERROR_COLOR, 410)
            draw_centered(screen, label_font, "Esc to go back", UNTYPED_COLOR, 470)

        # ---------------- LOBBY ----------------
        elif state == "LOBBY":
            draw_centered(screen, title_font, TITLE_TEXT, TITLE_COLOR, TITLE_Y)
            draw_centered(screen, button_font, "Share this code with your opponent", UNTYPED_COLOR, 240)
            box = pygame.Rect(0, 0, 360, 100)
            box.center = (cx, 330)
            pygame.draw.rect(screen, BAR_BG_COLOR, box, border_radius=12)
            code = code_font.render(net.room_code, True, P1_COLOR)
            screen.blit(code, code.get_rect(center=box.center))
            dots = "." * (int(time.time() * 2) % 4)
            draw_centered(screen, button_font, "Waiting for opponent" + dots, UNTYPED_COLOR, 430)
            draw_centered(screen, label_font, "Esc to cancel", UNTYPED_COLOR, 480)
            if net.status:
                draw_centered(screen, label_font, net.status, ERROR_COLOR, 520)

        # ---------------- GAME ----------------
        elif state == "GAME":
            now = time.time()
            playing = now >= game_start and winner is None

            # Webcam (top right)
            if cap is not None and cap.isOpened():
                ok, frame = cap.read()
                if ok:
                    if playing:
                        recognized = recognize_sign(frame, holistic, model)
                        if (recognized and progress[0] < len(game_text)
                                and recognized.lower() == game_text[progress[0]].lower()):
                            progress[0] += 1
                    webcam_surface = frame_to_surface(frame)
            cam_rect = pygame.Rect(SCREEN_WIDTH - WEBCAM_SIZE[0] - WEBCAM_MARGIN,
                                   WEBCAM_MARGIN, *WEBCAM_SIZE)
            if webcam_surface:
                screen.blit(webcam_surface, cam_rect)
            pygame.draw.rect(screen, WEBCAM_BORDER_COLOR, cam_rect, WEBCAM_BORDER)

            # Spaces have no sign, so skip over them
            while progress[0] < len(game_text) and game_text[progress[0]] == " ":
                progress[0] += 1

            # Hint for the letter the player is on (left of the webcam)
            if show_hints and winner is None and progress[0] < len(game_text):
                cur = game_text[progress[0]]
                hint_rect = pygame.Rect(cam_rect.x - HINT_GAP - HINT_SIZE[0],
                                        cam_rect.y, *HINT_SIZE)
                pygame.draw.rect(screen, HINT_BOX_COLOR, hint_rect, border_radius=12)
                img = hints.get(cur)
                if img:
                    screen.blit(img, img.get_rect(center=hint_rect.center))
                else:
                    msg = label_font.render("No hint image", True, UNTYPED_COLOR)
                    screen.blit(msg, msg.get_rect(center=hint_rect.center))
                tag = label_font.render(f"Sign: {cur.upper()}", True, P1_COLOR)
                screen.blit(tag, tag.get_rect(midtop=(hint_rect.centerx, hint_rect.bottom + 6)))

            # Sync with the database
            net.my_progress = progress[0]
            progress[1] = min(net.opp_progress, len(game_text))

            if winner is None:
                if progress[0] >= len(game_text):
                    winner = "You"
                elif progress[1] >= len(game_text):
                    winner = "Opponent"

            # Room code + progress bars
            screen.blit(label_font.render(f"Room: {net.room_code}", True, UNTYPED_COLOR),
                        (BAR_POS[0], BAR_POS[1] - 30))
            for i in range(2):
                y = BAR_POS[1] + 20 + i * BAR_SPACING
                screen.blit(label_font.render(PLAYER_NAMES[i], True, PLAYER_COLORS[i]), (BAR_POS[0], y))
                bar_bg = pygame.Rect(BAR_POS[0], y + 26, *BAR_SIZE)
                pygame.draw.rect(screen, BAR_BG_COLOR, bar_bg, border_radius=8)
                fill = bar_bg.copy()
                fill.width = int(BAR_SIZE[0] * progress[i] / len(game_text))
                pygame.draw.rect(screen, PLAYER_COLORS[i], fill, border_radius=8)

            # Text (bottom) with both players tracked
            for idx, (ch, x, y) in enumerate(char_layout):
                p1_done, p2_done = idx < progress[0], idx < progress[1]
                color = (BOTH_DONE_COLOR if p1_done and p2_done else
                         P1_COLOR if p1_done else
                         P2_COLOR if p2_done else UNTYPED_COLOR)
                screen.blit(glyph(ch, color), (TEXT_AREA.x + x, TEXT_AREA.y + y))

            # Carets, one per player
            for i in range(2):
                if progress[i] < len(char_layout):
                    ch, x, y = char_layout[progress[i]]
                    w = text_font.size(ch)[0]
                    cy = TEXT_AREA.y + y + text_font.get_height() + 2 + i * (CARET_THICKNESS + 2)
                    pygame.draw.rect(screen, PLAYER_COLORS[i],
                                     (TEXT_AREA.x + x, cy, w, CARET_THICKNESS))

            # Countdown / result overlays
            if now < game_start:
                n = int(game_start - now) + 1
                draw_centered(screen, big_font, str(n), TITLE_COLOR, SCREEN_HEIGHT // 2 - 40)
            elif winner:
                msg = "You win!" if winner == "You" else "Opponent wins!"
                draw_centered(screen, result_font, msg,
                              P1_COLOR if winner == "You" else P2_COLOR, 300)
                draw_centered(screen, label_font, "Esc for menu", UNTYPED_COLOR, 350)

            if net.status and "problem" in net.status:
                screen.blit(label_font.render(net.status, True, ERROR_COLOR),
                            (BAR_POS[0], SCREEN_HEIGHT - 30))

        elif state == "PRACTICE":
            now = time.time()
            cam_ok = False
            if cap is not None and cap.isOpened():
                ok, frame = cap.read()
                if ok:
                    cam_ok = True
                    webcam_surface = frame_to_surface(frame)
                    if now >= practice.flash_until:
                        detected = recognize_sign(frame, holistic, model)
                        practice.last_detected = detected or ""
                        if detected and detected.lower() == practice.target:
                            practice.correct()
            cam_rect = pygame.Rect(SCREEN_WIDTH - WEBCAM_SIZE[0] - WEBCAM_MARGIN,
                                   WEBCAM_MARGIN, *WEBCAM_SIZE)
            if webcam_surface and cam_ok:
                screen.blit(webcam_surface, cam_rect)
            pygame.draw.rect(screen, WEBCAM_BORDER_COLOR, cam_rect, WEBCAM_BORDER)

            # Webcam test info under the camera
            cam_text = f"Camera {cam_index} OK" if cam_ok else "No camera found"
            cam_surf = label_font.render(cam_text, True, P1_COLOR if cam_ok else ERROR_COLOR)
            screen.blit(cam_surf, cam_surf.get_rect(topright=(cam_rect.right, cam_rect.bottom + 76)))
            det = practice.last_detected.upper() if practice.last_detected else "-"
            det_surf = label_font.render(f"Detected: {det}", True, TITLE_COLOR)
            screen.blit(det_surf, det_surf.get_rect(topright=(cam_rect.right, cam_rect.bottom + 106)))

            # Stats (top left)
            avg = f"{practice.total_time / practice.score:.1f}s" if practice.score else "-"
            lines = [("Practice", P1_COLOR),
                     (f"Range: {practice.mode_name()}", UNTYPED_COLOR),
                     (f"Score: {practice.score}", TITLE_COLOR),
                     (f"Streak: {practice.streak}  (best {practice.best})", TITLE_COLOR),
                     (f"Avg time: {avg}", TITLE_COLOR)]
            for i, (text, color) in enumerate(lines):
                screen.blit(label_font.render(text, True, color),
                            (PRACTICE_STATS_POS[0], PRACTICE_STATS_POS[1] + i * PRACTICE_STATS_SPACING))

            # Hint box (uses the shared "Show hints" checkbox)
            if show_hints:
                draw_hint_box(screen, hints, practice.target, cam_rect, label_font)

            # Big target letter
            draw_centered(screen, button_font, "Sign this:", UNTYPED_COLOR, PRACTICE_TARGET_Y - 130)
            draw_centered(screen, target_font, practice.target.upper(), TITLE_COLOR, PRACTICE_TARGET_Y)
            if now < practice.flash_until:
                draw_centered(screen, button_font, practice.flash_text, practice.flash_color,
                              PRACTICE_TARGET_Y + 130)
            draw_centered(screen, label_font,
                          f"Enter: skip     Tab: change range ({practice.mode_name()})     Esc: menu",
                          UNTYPED_COLOR, SCREEN_HEIGHT - 25)

        cam_btn = camera_button_rect()
        if cam_btn:
            label = ("Searching..." if cam_state["switching"]
                     else f"{CAMERA_BUTTON_TEXT} ({cam_index})")
            draw_small_button(screen, label_font, label, cam_btn, not cam_state["switching"])
            if cam_state["msg"] and time.time() < cam_state["msg_until"]:
                note = label_font.render(cam_state["msg"], True, ERROR_COLOR)
                screen.blit(note, note.get_rect(topright=(cam_btn.right, cam_btn.bottom + 4)))

        cb = checkbox_rect()
        if cb:
            draw_checkbox(screen, label_font, HINT_CHECKBOX_TEXT, show_hints, cb.topleft)

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()