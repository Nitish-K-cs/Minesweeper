# minesweeper_bot_menu.py
# Requirements: pygame
# pip install pygame

import random
import sys
import time

import pygame
from collections import deque, defaultdict

# -------------------------
# Game logic (no pygame)
# -------------------------
class Minesweeper:
    HIDDEN = -1
    FLAG = -2
    MINE = -3

    def __init__(self, rows: int = 16, cols: int = 16, mines: int = 40) -> None:
        self.rows = rows
        self.cols = cols
        self.total_mines = mines
        self.reset()

    def reset(self) -> None:
        r, c = self.rows, self.cols
        # underlying board: numbers or MINE
        self.board = [[0] * c for _ in range(r)]
        # player view: HIDDEN, FLAG, or number (>=0)
        self.visible = [[Minesweeper.HIDDEN] * c for _ in range(r)]
        self.mines_placed = False
        self.game_over = False
        self.victory = False
        self.revealed_count = 0
        self.flags = 0

    def place_mines(self, safe_r: int, safe_c: int) -> None:
        # Place mines avoiding safe cell and its neighbors (first click safe)
        all_cells = [(i, j) for i in range(self.rows) for j in range(self.cols)]
        neighbors = [
            (safe_r + dr, safe_c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
        ]
        allowed = [p for p in all_cells if p not in neighbors]
        mines = random.sample(allowed, self.total_mines)

        for (i, j) in mines:
            self.board[i][j] = Minesweeper.MINE

        # fill numbers
        for i in range(self.rows):
            for j in range(self.cols):
                if self.board[i][j] == Minesweeper.MINE:
                    continue
                cnt = sum(
                    1 for (ni, nj) in self._neigh(i, j) if self.board[ni][nj] == Minesweeper.MINE
                )
                self.board[i][j] = cnt

        self.mines_placed = True

    def _neigh(self, r: int, c: int):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    yield nr, nc

    def open(self, r: int, c: int) -> None:
        if self.game_over or self.visible[r][c] == self.visible_flag():
            return

        if not self.mines_placed:
            self.place_mines(r, c)

        if self.visible[r][c] == Minesweeper.FLAG:
            return

        if self.board[r][c] == Minesweeper.MINE:
            # reveal mine -> game over
            self.visible[r][c] = self.board[r][c]
            self.game_over = True
            self.victory = False
            return

        # BFS reveal
        q = deque()
        if self.visible[r][c] == Minesweeper.HIDDEN:
            q.append((r, c))

        while q:
            i, j = q.popleft()
            if self.visible[i][j] != Minesweeper.HIDDEN:
                continue
            self.visible[i][j] = self.board[i][j]
            self.revealed_count += 1
            if self.board[i][j] == 0:
                for (ni, nj) in self._neigh(i, j):
                    if self.visible[ni][nj] == Minesweeper.HIDDEN:
                        q.append((ni, nj))

        self._check_victory()

    def flag(self, r: int, c: int) -> None:
        if self.game_over:
            return

        if self.visible[r][c] == Minesweeper.HIDDEN:
            self.visible[r][c] = Minesweeper.FLAG
            self.flags += 1
        elif self.visible[r][c] == Minesweeper.FLAG:
            self.visible[r][c] = Minesweeper.HIDDEN
            self.flags -= 1

        self._check_victory()

    def visible_flag(self) -> int:
        return Minesweeper.FLAG

    def _check_victory(self) -> None:
        # win when all non-mine cells revealed
        to_reveal = self.rows * self.cols - self.total_mines
        if self.revealed_count == to_reveal and not self.game_over:
            self.game_over = True
            self.victory = True

# -------------------------
# Bot logic
# -------------------------
class MinesweeperBot:
    def __init__(self, game: Minesweeper, enum_limit: int = 20) -> None:
        self.game = game
        self.enum_limit = enum_limit  # boundary size limit for exact enumeration

    def step(self):
        """Perform one reasoning step. Return action or None if no action."""
        g = self.game

        # 1) deterministic moves
        made_action = self._deterministic_pass()
        if made_action:
            return made_action

        # 2) boundary reasoning: compute exact probabilities if possible
        boundary, constraints = self._build_boundary_and_constraints()
        if boundary and len(boundary) <= self.enum_limit:
            probs = self._enumerate_probs(boundary, constraints)
            # pick cell with minimal mine probability to open (if any < 1)
            best_cell, best_prob = min(probs.items(), key=lambda kv: kv[1])
            if best_prob == 1.0:
                r, c = best_cell
                g.flag(r, c)
                return ("flag", best_cell)
            r, c = best_cell
            g.open(r, c)
            return ("open", best_cell, best_prob)

        # 3) fallback heuristic: pick an unknown with minimal global prob
        cell = self._fallback_pick()
        if cell is not None:
            g.open(*cell)
            return ("open", cell, None)

        return None

    def _deterministic_pass(self) -> bool:
        g = self.game
        # iterate over all revealed numbered cells
        for i in range(g.rows):
            for j in range(g.cols):
                if g.visible[i][j] >= 0:  # revealed number
                    num = g.visible[i][j]
                    neigh = list(g._neigh(i, j))
                    flagged = sum(1 for (a, b) in neigh if g.visible[a][b] == Minesweeper.FLAG)
                    hidden = [(a, b) for (a, b) in neigh if g.visible[a][b] == Minesweeper.HIDDEN]

                    # rule 1: if flagged == num -> open all hidden neighbors
                    if flagged == num and hidden:
                        for (a, b) in hidden:
                            g.open(a, b)
                        return True

                    # rule 2: if len(hidden) == num - flagged -> flag all hidden
                    if hidden and len(hidden) == (num - flagged):
                        for (a, b) in hidden:
                            g.flag(a, b)
                        return True

        return False

    def _build_boundary_and_constraints(self):
        g = self.game
        boundary_set = set()
        constraints = []  # each constraint: (cells_list, mines_required)

        for i in range(g.rows):
            for j in range(g.cols):
                if g.visible[i][j] >= 0:
                    num = g.visible[i][j]
                    neigh = list(g._neigh(i, j))
                    hidden = [
                        (a, b) for (a, b) in neigh if g.visible[a][b] == Minesweeper.HIDDEN
                    ]
                    flagged = sum(1 for (a, b) in neigh if g.visible[a][b] == Minesweeper.FLAG)

                    if hidden:
                        mines_needed = max(0, num - flagged)
                        constraints.append((hidden, mines_needed))
                        for cell in hidden:
                            boundary_set.add(cell)

        # boundary is only the hidden cells that appear in constraints
        boundary = sorted(boundary_set)
        return boundary, constraints

    def _enumerate_probs(self, boundary, constraints):
        """Enumerate all assignments on boundary consistent with constraints.
        Return dict mapping cell -> probability of mine (0..1).
        """

        g = self.game
        bindex = {cell: i for i, cell in enumerate(boundary)}
        n = len(boundary)
        counts = [0] * n
        total_valid = 0

        # enumerate bitmasks
        for mask in range(1 << n):
            ok = True
            for (cells, need) in constraints:
                s = 0
                for (a, b) in cells:
                    if ((mask >> bindex[(a, b)]) & 1):
                        s += 1
                if s != need:
                    ok = False
                    break
            if not ok:
                continue

            total_valid += 1
            for i in range(n):
                if (mask >> i) & 1:
                    counts[i] += 1

        if total_valid == 0:
            # no valid assignment found; return uniform estimate
            unknowns = [
                (i, j)
                for i in range(g.rows)
                for j in range(g.cols)
                if g.visible[i][j] == Minesweeper.HIDDEN
            ]
            if not unknowns:
                return {cell: 1.0 for cell in boundary}
            prob = (g.total_mines - g.flags) / len(unknowns)
            return {cell: prob for cell in boundary}

        probs = {boundary[i]: counts[i] / total_valid for i in range(n)}
        return probs

    def _fallback_pick(self):
        # pick an unknown cell with lowest estimated risk
        g = self.game
        unknowns = [
            (i, j) for i in range(g.rows) for j in range(g.cols) if g.visible[i][j] == Minesweeper.HIDDEN
        ]
        if not unknowns:
            return None

        remaining_mines = g.total_mines - g.flags
        # simple global prob (not used directly here, but kept for potential extensions)
        _global_prob = remaining_mines / len(unknowns) if unknowns else 1.0

        # try to pick a cell not adjacent to revealed numbers first (safer heuristic)
        safe_candidates = [
            cell
            for cell in unknowns
            if all(g.visible[a][b] == Minesweeper.HIDDEN for a, b in g._neigh(*cell))
        ]
        pick_list = safe_candidates if safe_candidates else unknowns
        return random.choice(pick_list)

# -------------------------
# Pygame UI
# -------------------------
CELL = 36
MARGIN = 6
TOP_MARGIN = 80
PANEL_WIDTH = 300
BOARD_OFFSET_X = 28

COLORS = {
    'bg': (16, 24, 48),            # deep navy background
    'panel': (20, 26, 60),         # slightly lighter panel
    'cell_hidden': (72, 160, 240), # bright sky blue for hidden cells
    'cell_revealed': (250, 250, 255),
    'cell_flag': (255, 90, 90),    # vivid red flag
    'text': (30, 36, 52),
    'mine': (28, 24, 24),
    'grid': (200, 215, 235),       # light grid lines
    'button': (98, 58, 218),       # purple button
    'button_hover': (140, 100, 255),
    'button_text': (255, 255, 255),
    'highlight': (255, 255, 255, 72),
    'info_bg': (40, 44, 90),        # darker rounded badge behind info
    'info_text': (220, 230, 255),
    'popup_bg': (255, 200, 64),    # bright popup background
    'popup_text': (18, 18, 28)
}

NUM_COLORS = [
    (0, 0, 0),            # 0 (unused)
    (40, 150, 255),       # 1 - bright blue
    (80, 220, 120),       # 2 - neon green
    (255, 120, 60),       # 3 - orange
    (150, 120, 255),      # 4 - light purple
    (255, 90, 140),       # 5 - pink
    (0, 200, 200),        # 6 - cyan
    (80, 80, 90),         # 7 - muted
    (120, 120, 140)       # 8 - subtle gray-blue
]

DIFFICULTIES = {
    "8x8":  (8, 8, 10),   # rows, cols, mines
    "16x16": (16, 16, 40)
}

def draw_board(screen, font, game, offset_x):
    # background for board area only (panel covers rest)
    screen.fill(COLORS['bg'])
    rows, cols = game.rows, game.cols
    # draw info
    info = f"Mines: {game.total_mines}  Flags: {game.flags}  Revealed: {game.revealed_count}/{game.rows*game.cols - game.total_mines}"
    text = font.render(info, True, COLORS['info_text'])
    # draw info (centered badge above the board)
    board_w = cols * CELL + (cols + 1) * MARGIN
    info_pad_x, info_pad_y = 14, 8
    info_w = text.get_width() + info_pad_x * 2
    info_h = text.get_height() + info_pad_y * 2
    info_x = offset_x + board_w // 2 - info_w // 2
    info_y = 18
    info_rect = pygame.Rect(info_x, info_y, info_w, info_h)
    pygame.draw.rect(screen, COLORS['info_bg'], info_rect, border_radius=10)
    pygame.draw.rect(screen, COLORS['button'], info_rect, 2, border_radius=10)
    screen.blit(text, (info_x + info_pad_x, info_y + info_pad_y))
    mx, my = pygame.mouse.get_pos()
    # draw cells
    for i in range(rows):
        for j in range(cols):
            x = offset_x + MARGIN + j*(CELL+MARGIN)
            y = TOP_MARGIN + MARGIN + i*(CELL+MARGIN)
            rect = pygame.Rect(x,y,CELL,CELL)
            val = game.visible[i][j]
            # hover highlight
            hover = rect.collidepoint(mx, my)
            if val == Minesweeper.HIDDEN:
                pygame.draw.rect(screen, COLORS['cell_hidden'], rect, border_radius=6)
                if hover:
                    pygame.draw.rect(screen, (255,255,255), rect, 2, border_radius=6)
            elif val == Minesweeper.FLAG:
                pygame.draw.rect(screen, COLORS['cell_revealed'], rect, border_radius=6)
                draw_flag(screen, rect)
            elif val == Minesweeper.MINE:
                pygame.draw.rect(screen, COLORS['cell_revealed'], rect, border_radius=6)
                draw_mine(screen, rect)
            else:
                pygame.draw.rect(screen, COLORS['cell_revealed'], rect, border_radius=6)
                if val > 0:
                    txt = font.render(str(val), True, NUM_COLORS[min(val,len(NUM_COLORS)-1)])
                    screen.blit(txt, (x + (CELL - txt.get_width())//2, y + (CELL - txt.get_height())//2))
            pygame.draw.rect(screen, COLORS['grid'], rect, 1, border_radius=6)
    # game over popup
    if game.game_over:
        s = "You win! (press click to restart)" if game.victory else "Game over (press click to restart)"
        # dark translucent overlay
        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        overlay.fill((8, 8, 12, 160))
        screen.blit(overlay, (0, 0))

        # popup box
        msg = font.render(s, True, COLORS['popup_text'])
        pad_x, pad_y = 28, 18
        popup_w = msg.get_width() + pad_x * 2
        popup_h = msg.get_height() + pad_y * 2
        sw, sh = screen.get_size()
        popup_x = sw // 2 - popup_w // 2
        popup_y = sh // 2 - popup_h // 2

        popup_rect = pygame.Rect(popup_x, popup_y, popup_w, popup_h)
        pygame.draw.rect(screen, COLORS['popup_bg'], popup_rect, border_radius=12)
        # subtle border
        pygame.draw.rect(screen, COLORS['button'], popup_rect, 3, border_radius=12)

        # blit message centered in popup
        screen.blit(msg, (popup_x + pad_x, popup_y + pad_y))


def draw_flag(screen, rect):
    # small triangle flag with staff
    cx = rect.x + 8
    cy = rect.y + 6
    staff_top = (cx, rect.y + 6)
    staff_bottom = (cx, rect.y + rect.height - 6)
    pygame.draw.line(screen, (40,40,40), staff_top, staff_bottom, 3)
    # flag triangle
    points = [(cx+2, rect.y + 8), (cx+18, rect.y + 12), (cx+2, rect.y + 18)]
    pygame.draw.polygon(screen, COLORS['cell_flag'], points)
    pygame.draw.polygon(screen, (30,30,30), points, 1)


def draw_mine(screen, rect):
    cx = rect.x + rect.width//2
    cy = rect.y + rect.height//2
    r = rect.width//3
    pygame.draw.circle(screen, COLORS['mine'], (cx, cy), r)
    pygame.draw.circle(screen, (200,200,200), (cx - r//3, cy - r//3), r//5)

def draw_button(screen, rect, text, font, hover=False):
    color = COLORS['button_hover'] if hover else COLORS['button']
    pygame.draw.rect(screen, color, rect, border_radius=8)
    label = font.render(text, True, COLORS['button_text'])
    lx = rect.x + (rect.width - label.get_width()) // 2
    ly = rect.y + (rect.height - label.get_height()) // 2
    screen.blit(label, (lx, ly))

def draw_menu(screen, title_font, font, difficulty, button_rects):
    screen.fill(COLORS['bg'])
    width, height = screen.get_size()

    title = title_font.render("Minesweeper + Bot", True, (240,240,240))
    screen.blit(title, (width//2 - title.get_width()//2, 60))

    start_rect, diff_rect, quit_rect = button_rects

    mx, my = pygame.mouse.get_pos()

    draw_button(screen, start_rect, "Start", font, start_rect.collidepoint(mx,my))
    draw_button(screen, diff_rect, f"Difficulty: {difficulty}", font, diff_rect.collidepoint(mx,my))
    draw_button(screen, quit_rect, "Quit", font, quit_rect.collidepoint(mx,my))

    hint = font.render("Left click buttons | SPACE: bot auto-play in game", True, (200,200,200))
    screen.blit(hint, (width//2 - hint.get_width()//2, height - 40))

def draw_side_panel(screen, font, panel_rect, game, history, auto_play):
    pygame.draw.rect(screen, (35,35,35), panel_rect)

    x = panel_rect.x + 15
    y = panel_rect.y + 15

    title = font.render("CONTROL PANEL", True, (255,255,255))
    screen.blit(title, (x, y))
    y += 35

    buttons = {
        "bot": pygame.Rect(x, y, 260, 40),
        "restart": pygame.Rect(x, y+50, 260, 40),
        "menu": pygame.Rect(x, y+100, 260, 40),
        "settings": pygame.Rect(x, y+150, 260, 40),
    }

    for name, rect in buttons.items():
        draw_button(
            screen,
            rect,
            f"{'Stop' if auto_play else 'Start'} Bot" if name == "bot" else name.capitalize(),
            font,
            rect.collidepoint(pygame.mouse.get_pos())
        )

    y += 210
    hist_title = font.render("GAME HISTORY", True, (255,255,255))
    screen.blit(hist_title, (x, y))
    y += 30

    for h in history[-8:]:
        txt = font.render(h, True, (200,200,200))
        screen.blit(txt, (x, y))
        y += 22

    return buttons


def main():
    pygame.init()

    # We size the window for the largest board (16x16) plus side panel
    max_rows, max_cols = 16, 16
    board_w = max_cols*CELL + (max_cols+1)*MARGIN
    board_h = TOP_MARGIN + max_rows*CELL + (max_rows+1)*MARGIN
    width = BOARD_OFFSET_X*2 + board_w + PANEL_WIDTH
    height = max(board_h, 480)

    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    SCREEN_W, SCREEN_H = screen.get_size()
    pygame.display.set_caption("Minesweeper + Bot")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    title_font = pygame.font.SysFont("consolas", 32, bold=True)

    # Menu state
    state = "menu"  # "menu" or "game"
    difficulty = "16x16"  # default
    game = None
    bot = None
    auto_play = False
    step_delay = 0.05  # seconds between bot steps
    last_step = time.time()
    game_history = []
    game_start_time = time.time()

    # Button layout for menu
    BUTTON_W, BUTTON_H = 240, 52
    start_rect = pygame.Rect(SCREEN_W//2 - BUTTON_W//2, 160, BUTTON_W, BUTTON_H)
    diff_rect = pygame.Rect(SCREEN_W//2 - BUTTON_W//2, 230, BUTTON_W, BUTTON_H)
    quit_rect = pygame.Rect(SCREEN_W//2 - BUTTON_W//2, 300, BUTTON_W, BUTTON_H)
    button_rects = (start_rect, diff_rect, quit_rect)

    running = True
    while running:
        # current window size and dynamic board offset (used for centering small boards)
        cur_w, cur_h = screen.get_size()
        offset_x = BOARD_OFFSET_X
        if state == "game" and game is not None:
            board_area_width = cur_w - PANEL_WIDTH - 2 * BOARD_OFFSET_X
            board_w = game.cols * CELL + (game.cols + 1) * MARGIN
            offset_x = BOARD_OFFSET_X + max(0, (board_area_width - board_w) // 2)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if state == "menu":
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    if start_rect.collidepoint(mx,my):
                        # Create game with selected difficulty
                        rows, cols, mines = DIFFICULTIES[difficulty]
                        game = Minesweeper(rows, cols, mines)
                        bot = MinesweeperBot(game, enum_limit=20)
                        auto_play = False
                        state = "game"
                    elif diff_rect.collidepoint(mx,my):
                        # Toggle difficulty
                        difficulty = "8x8" if difficulty == "16x16" else "16x16"
                    elif quit_rect.collidepoint(mx,my):
                        pygame.quit(); sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        pygame.quit(); sys.exit()
            elif state == "game":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos

                    # ================= STEP 10: RIGHT PANEL BUTTON HANDLING =================
                    panel_rect = pygame.Rect(cur_w - PANEL_WIDTH, 0, PANEL_WIDTH, cur_h)

                    if panel_rect.collidepoint(mx, my):
                        # buttons may not exist yet during the first event loop iteration;
                        # recreate their rects to safely test clicks.
                        if 'buttons' not in locals() or buttons is None:
                            x = panel_rect.x + 15
                            y = panel_rect.y + 15 + 35
                            buttons = {
                                "bot": pygame.Rect(x, y, 260, 40),
                                "restart": pygame.Rect(x, y+50, 260, 40),
                                "menu": pygame.Rect(x, y+100, 260, 40),
                                "settings": pygame.Rect(x, y+150, 260, 40),
                            }

                        if buttons["bot"].collidepoint(mx, my):
                            auto_play = not auto_play

                        elif buttons["restart"].collidepoint(mx, my):
                            game.reset()
                            bot = MinesweeperBot(game, enum_limit=20)
                            game_start_time = time.time()

                        elif buttons["menu"].collidepoint(mx, my):
                            state = "menu"
                            game = None
                            bot = None
                            auto_play = False

                        elif buttons["settings"].collidepoint(mx, my):
                            print("Settings toggle (future feature)")

                        continue
                    # =======================================================================

                    if game.game_over:
                        # restart current difficulty on click
                        game.reset()
                        bot = MinesweeperBot(game, enum_limit=20)
                        game_start_time = time.time()
                    else:
                        if my > TOP_MARGIN:
                            j = (mx - offset_x - MARGIN) // (CELL + MARGIN)
                            i = (my - TOP_MARGIN - MARGIN) // (CELL + MARGIN)
                            if 0 <= i < game.rows and 0 <= j < game.cols:
                                if event.button == 1:
                                    game.open(i, j)
                                elif event.button == 3:
                                    game.flag(i, j)

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        auto_play = not auto_play

                    elif event.key == pygame.K_r:
                        game.reset()
                        bot = MinesweeperBot(game, enum_limit=20)
                        game_start_time = time.time()

                    elif event.key == pygame.K_RIGHT:
                        # single bot step
                        action = bot.step()

                    elif event.key == pygame.K_UP:
                        bot.enum_limit += 1
                        print("enum_limit:", bot.enum_limit)

                    elif event.key == pygame.K_DOWN:
                        bot.enum_limit = max(5, bot.enum_limit - 1)
                        print("enum_limit:", bot.enum_limit)

                    elif event.key == pygame.K_ESCAPE:
                        # go back to menu
                        state = "menu"
                        game = None
                        bot = None
                        auto_play = False


        # Update / draw
        if state == "menu":
            draw_menu(screen, title_font, font, difficulty, button_rects)
        else:  # game
            # auto-play
            if auto_play and game is not None and not game.game_over and (time.time() - last_step > step_delay):
                action = bot.step()
                last_step = time.time()

            panel_rect = pygame.Rect(cur_w - PANEL_WIDTH, 0, PANEL_WIDTH, cur_h)

            buttons = draw_side_panel(screen, font, panel_rect, game, game_history, auto_play)
            draw_board(screen, font, game, offset_x)

        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    main()
