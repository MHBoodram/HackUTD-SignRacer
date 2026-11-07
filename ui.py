import random
import string
import sys

import cv2
import pygame

# ============================ CONFIG ============================
# --- Window ---
SCREEN_WIDTH  = 1280
SCREEN_HEIGHT = 720
FPS           = 60
WINDOW_CAPTION = "SignRacer"

# --- Title ---
TITLE_TEXT       = "SignRacer"
TITLE_FONT_SIZE  = 96
TITLE_COLOR      = (255, 255, 255)
TITLE_Y          = 90            # vertical center of the title on the menu

# --- Colors ---
BG_COLOR         = (24, 26, 36)
BUTTON_COLOR     = (70, 90, 200)
BUTTON_HOVER     = (100, 120, 235)
BUTTON_TEXT_COLOR = (255, 255, 255)
UNTYPED_COLOR    = (120, 125, 145)   # letters nobody has signed yet
BOTH_DONE_COLOR  = (200, 200, 200)   # letters both players have passed
P1_COLOR         = (80, 200, 120)    # you
P2_COLOR         = (240, 120, 90)    # opponent

# --- Menu buttons ---
BUTTON_WIDTH     = 360
BUTTON_HEIGHT    = 80
BUTTON_SPACING   = 30
BUTTON_FONT_SIZE = 36
BUTTON_START_Y   = 260             # y of the first button
CREATE_BUTTON_TEXT = "Create Room"
JOIN_BUTTON_TEXT   = "Join with Code"

# --- Room code ---
ROOM_CODE_LENGTH = 5
ROOM_CODE_CHARS  = string.ascii_uppercase + string.digits
CODE_FONT_SIZE   = 48

# --- Webcam (top right) ---
WEBCAM_INDEX     = 0
WEBCAM_SIZE      = (360, 270)      # (width, height) on screen
WEBCAM_MARGIN    = 20              # gap from top/right edges
WEBCAM_MIRROR    = True
WEBCAM_BORDER    = 3
WEBCAM_BORDER_COLOR = (255, 255, 255)

# --- Typing text (bottom) ---
TEXT_FONT_SIZE   = 40
TEXT_LINE_SPACING = 64
TEXT_AREA        = pygame.Rect(40, 380, SCREEN_WIDTH - 80, 300)  # x, y, w, h
GAME_TEXT        = "the quick brown fox jumps over the lazy dog"
CARET_THICKNESS  = 4

# --- Player labels / progress bars (top left of game screen) ---
PLAYER_NAMES     = ["You", "Opponent"]
PLAYER_COLORS    = [P1_COLOR, P2_COLOR]
LABEL_FONT_SIZE  = 28
BAR_POS          = (40, 40)        # top-left of first bar
BAR_SIZE         = (500, 24)
BAR_SPACING      = 60
BAR_BG_COLOR     = (50, 54, 70)

# --- Testing ---
DEBUG_KEYBOARD   = True            # type on keyboard to advance player 1
# ================================================================


class Button:
    def __init__(self, text, center, font):
        self.text = text
        self.font = font
        self.rect = pygame.Rect(0, 0, BUTTON_WIDTH, BUTTON_HEIGHT)
        self.rect.center = center

    def draw(self, surf):
        hovered = self.rect.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(surf, BUTTON_HOVER if hovered else BUTTON_COLOR,
                         self.rect, border_radius=16)
        label = self.font.render(self.text, True, BUTTON_TEXT_COLOR)
        surf.blit(label, label.get_rect(center=self.rect.center))

    def clicked(self, event):
        return (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.rect.collidepoint(event.pos))


def generate_room_code():
    return "".join(random.choices(ROOM_CODE_CHARS, k=ROOM_CODE_LENGTH))


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
    return positions[:-1]  # drop trailing space


def frame_to_surface(frame):
    frame = cv2.resize(frame, WEBCAM_SIZE)
    if WEBCAM_MIRROR:
        frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(frame.tobytes(), WEBCAM_SIZE, "RGB")


def recognize_sign(frame):
    """HOOK: plug your ASL model in here. Return a letter (str) or None."""
    return None


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(WINDOW_CAPTION)
    clock = pygame.time.Clock()

    title_font  = pygame.font.Font(None, TITLE_FONT_SIZE)
    button_font = pygame.font.Font(None, BUTTON_FONT_SIZE)
    code_font   = pygame.font.Font(None, CODE_FONT_SIZE)
    label_font  = pygame.font.Font(None, LABEL_FONT_SIZE)
    text_font   = pygame.font.SysFont("couriernew", TEXT_FONT_SIZE, bold=True)

    cx = SCREEN_WIDTH // 2
    create_btn = Button(CREATE_BUTTON_TEXT, (cx, BUTTON_START_Y + BUTTON_HEIGHT // 2), button_font)
    join_btn = Button(JOIN_BUTTON_TEXT,
                      (cx, BUTTON_START_Y + BUTTON_HEIGHT * 3 // 2 + BUTTON_SPACING), button_font)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    char_layout = layout_text(GAME_TEXT, text_font, TEXT_AREA.width)

    state = "MENU"             # MENU, JOIN, GAME
    room_code = ""
    typed_code = ""
    progress = [0, 0]          # index into GAME_TEXT for each player (0 = you, 1 = opponent)
    webcam_surface = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                sys.exit()

            if state == "MENU":
                if create_btn.clicked(event):
                    room_code = generate_room_code()
                    progress = [0, 0]
                    state = "GAME"          # TODO: start your server/room here
                elif join_btn.clicked(event):
                    typed_code = ""
                    state = "JOIN"

            elif state == "JOIN":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        state = "MENU"
                    elif event.key == pygame.K_BACKSPACE:
                        typed_code = typed_code[:-1]
                    elif event.key == pygame.K_RETURN and len(typed_code) == ROOM_CODE_LENGTH:
                        room_code = typed_code
                        progress = [0, 0]
                        state = "GAME"      # TODO: connect to the room here
                    elif event.unicode.upper() in ROOM_CODE_CHARS and len(typed_code) < ROOM_CODE_LENGTH:
                        typed_code += event.unicode.upper()

            elif state == "GAME":
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    state = "MENU"
                elif DEBUG_KEYBOARD and event.type == pygame.KEYDOWN:
                    if progress[0] < len(GAME_TEXT) and event.unicode == GAME_TEXT[progress[0]]:
                        progress[0] += 1

        screen.fill(BG_COLOR)

        # ---------------- MENU ----------------
        if state == "MENU":
            t = title_font.render(TITLE_TEXT, True, TITLE_COLOR)
            screen.blit(t, t.get_rect(center=(cx, TITLE_Y)))
            create_btn.draw(screen)
            join_btn.draw(screen)

        # ---------------- JOIN ----------------
        elif state == "JOIN":
            t = title_font.render(TITLE_TEXT, True, TITLE_COLOR)
            screen.blit(t, t.get_rect(center=(cx, TITLE_Y)))
            prompt = button_font.render("Enter room code, then press Enter", True, UNTYPED_COLOR)
            screen.blit(prompt, prompt.get_rect(center=(cx, 240)))
            box = pygame.Rect(0, 0, 360, 80)
            box.center = (cx, 330)
            pygame.draw.rect(screen, BAR_BG_COLOR, box, border_radius=12)
            code = code_font.render(typed_code, True, TITLE_COLOR)
            screen.blit(code, code.get_rect(center=box.center))

        # ---------------- GAME ----------------
        elif state == "GAME":
            # Webcam (top right)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok:
                    recognized = recognize_sign(frame)
                    if (recognized and progress[0] < len(GAME_TEXT)
                            and recognized.lower() == GAME_TEXT[progress[0]].lower()):
                        progress[0] += 1
                    webcam_surface = frame_to_surface(frame)
            cam_rect = pygame.Rect(SCREEN_WIDTH - WEBCAM_SIZE[0] - WEBCAM_MARGIN,
                                   WEBCAM_MARGIN, *WEBCAM_SIZE)
            if webcam_surface:
                screen.blit(webcam_surface, cam_rect)
            pygame.draw.rect(screen, WEBCAM_BORDER_COLOR, cam_rect, WEBCAM_BORDER)

            # Room code + progress bars (top left)
            screen.blit(label_font.render(f"Room: {room_code}", True, UNTYPED_COLOR),
                        (BAR_POS[0], BAR_POS[1] - 30))
            for i in range(2):
                y = BAR_POS[1] + 20 + i * BAR_SPACING
                screen.blit(label_font.render(PLAYER_NAMES[i], True, PLAYER_COLORS[i]), (BAR_POS[0], y))
                bar_bg = pygame.Rect(BAR_POS[0], y + 26, *BAR_SIZE)
                pygame.draw.rect(screen, BAR_BG_COLOR, bar_bg, border_radius=8)
                fill = bar_bg.copy()
                fill.width = int(BAR_SIZE[0] * progress[i] / len(GAME_TEXT))
                pygame.draw.rect(screen, PLAYER_COLORS[i], fill, border_radius=8)

            # Text (bottom) with both players tracked
            for idx, (ch, x, y) in enumerate(char_layout):
                p1_done, p2_done = idx < progress[0], idx < progress[1]
                color = (BOTH_DONE_COLOR if p1_done and p2_done else
                         P1_COLOR if p1_done else
                         P2_COLOR if p2_done else UNTYPED_COLOR)
                screen.blit(text_font.render(ch, True, color),
                            (TEXT_AREA.x + x, TEXT_AREA.y + y))

            # Carets: one per player, stacked so they don't overlap
            for i in range(2):
                if progress[i] < len(char_layout):
                    ch, x, y = char_layout[progress[i]]
                    w = text_font.size(ch)[0]
                    cy = TEXT_AREA.y + y + text_font.get_height() + 2 + i * (CARET_THICKNESS + 2)
                    pygame.draw.rect(screen, PLAYER_COLORS[i],
                                     (TEXT_AREA.x + x, cy, w, CARET_THICKNESS))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()