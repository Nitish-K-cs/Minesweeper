# minesweeper_bot_menu.py
# Requirements: pygame
# pip install pygame

import pygame, sys, random, itertools, time
from collections import deque, defaultdict

# -------------------------
# Game logic (no pygame)
# -------------------------
class Minesweeper:
    HIDDEN = -1
    FLAG = -2
    MINE = -3

    def __init__(self, rows=16, cols=16, mines=40):
        self.rows = rows
        self.cols = cols
        self.total_mines = mines
        self.reset()

    def reset(self):
        r, c = self.rows, self.cols
        self.board = [[0]*c for _ in range(r)]         # underlying board: numbers or -3 for mine
        self.visible = [[Minesweeper.HIDDEN]*c for _ in range(r)]  # player view: HIDDEN, FLAG, or number (>=0)
        self.mines_placed = False
        self.game_over = False
        self.victory = False
        self.revealed_count = 0
        self.flags = 0

    def place_mines(self, safe_r, safe_c):
        # Place mines avoiding safe cell and its neighbors (first click safe)
        all_cells = [(i,j) for i in range(self.rows) for j in range(self.cols)]
        neighbors = [(safe_r+dr, safe_c+dc) for dr in (-1,0,1) for dc in (-1,0,1)]
        allowed = [p for p in all_cells if p not in neighbors]
        mines = random.sample(allowed, self.total_mines)
        for (i,j) in mines:
            self.board[i][j] = Minesweeper.MINE
        # fill numbers
        for i in range(self.rows):
            for j in range(self.cols):
                if self.board[i][j] == Minesweeper.MINE: continue
                cnt = sum(1 for (ni,nj) in self._neigh(i,j) if self.board[ni][nj] == Minesweeper.MINE)
                self.board[i][j] = cnt
        self.mines_placed = True

    def _neigh(self, r, c):
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==0 and dc==0: continue
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    yield nr, nc

    def open(self, r, c):
        if self.game_over or self.visible[r][c] == self.visible_flag(): return
        if not self.mines_placed:
            self.place_mines(r, c)
        if self.visible[r][c] == Minesweeper.FLAG: return
        if self.board[r][c] == Minesweeper.MINE:
            # reveal mine -> game over
            self.visible[r][c] = self.board[r][c]
            self.game_over = True
            self.victory = False
            return
        # BFS reveal
        q = deque()
        if self.visible[r][c] == Minesweeper.HIDDEN:
            q.append((r,c))
        while q:
            i,j = q.popleft()
            if self.visible[i][j] != Minesweeper.HIDDEN: continue
            self.visible[i][j] = self.board[i][j]
            self.revealed_count += 1
            if self.board[i][j] == 0:
                for (ni,nj) in self._neigh(i,j):
                    if self.visible[ni][nj] == Minesweeper.HIDDEN:
                        q.append((ni,nj))
        self._check_victory()

    def flag(self, r, c):
        if self.game_over: return
        if self.visible[r][c] == Minesweeper.HIDDEN:
            self.visible[r][c] = Minesweeper.FLAG
            self.flags += 1
        elif self.visible[r][c] == Minesweeper.FLAG:
            self.visible[r][c] = Minesweeper.HIDDEN
            self.flags -= 1
        self._check_victory()

    def visible_flag(self):
        return Minesweeper.FLAG

    def _check_victory(self):
        # win when all non-mine cells revealed
        to_reveal = self.rows*self.cols - self.total_mines
        if self.revealed_count == to_reveal and not self.game_over:
            self.game_over = True
            self.victory = True

# -------------------------
# Bot logic
# -------------------------
class MinesweeperBot:
    def __init__(self, game: Minesweeper, enum_limit=20):
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
        if boundary:
            if len(boundary) <= self.enum_limit:
                probs = self._enumerate_probs(boundary, constraints)
                # pick cell with minimal mine probability to open (if any < 1)
                best_cell, best_prob = min(probs.items(), key=lambda kv: kv[1])
                if best_prob == 1.0:
                    # flag it
                    r,c = best_cell
                    g.flag(r,c)
                    return ('flag', best_cell)
                else:
                    r,c = best_cell
                    g.open(r,c)
                    return ('open', best_cell, best_prob)
            # else fallthrough to heuristic

        # 3) fallback heuristic: pick an unknown with minimal global prob
        cell = self._fallback_pick()
        if cell is not None:
            g.open(*cell)
            return ('open', cell, None)
        return None

    def _deterministic_pass(self):
        g = self.game
        # iterate over all revealed numbered cells
        for i in range(g.rows):
            for j in range(g.cols):
                if g.visible[i][j] >= 0:  # revealed number
                    num = g.visible[i][j]
                    neigh = list(g._neigh(i,j))
                    flagged = sum(1 for (a,b) in neigh if g.visible[a][b] == Minesweeper.FLAG)
                    hidden = [(a,b) for (a,b) in neigh if g.visible[a][b] == Minesweeper.HIDDEN]
                    # rule 1: if flagged == num -> open all hidden neighbors
                    if flagged == num and hidden:
                        for (a,b) in hidden:
                            g.open(a,b)
                        return True
                    # rule 2: if len(hidden) == num - flagged -> flag all hidden
                    if len(hidden) > 0 and len(hidden) == (num - flagged):
                        for (a,b) in hidden:
                            g.flag(a,b)
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
                    neigh = list(g._neigh(i,j))
                    hidden = [(a,b) for (a,b) in neigh if g.visible[a][b] == Minesweeper.HIDDEN]
                    flagged = sum(1 for (a,b) in neigh if g.visible[a][b] == Minesweeper.FLAG)
                    if hidden:
                        mines_needed = num - flagged
                        if mines_needed < 0:
                            mines_needed = 0
                        constraints.append((hidden, mines_needed))
                        for cell in hidden:
                            boundary_set.add(cell)
        # boundary is only the hidden cells that appear in constraints
        boundary = sorted(boundary_set)
        return boundary, constraints

    def _enumerate_probs(self, boundary, constraints):
        """Enumerate all assignments on boundary consistent with constraints.
           Return dict mapping cell -> probability of mine (0..1)."""
        g = self.game  # <-- FIX: need game reference here
        bindex = {cell:i for i,cell in enumerate(boundary)}
        n = len(boundary)
        counts = [0]*n
        total_valid = 0
        # enumerate bitmasks
        for mask in range(1<<n):
            ok = True
            for (cells, need) in constraints:
                s = 0
                for (a,b) in cells:
                    if ((mask >> bindex[(a,b)]) & 1):
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
            # no valid assignment found; return uniform
            unknowns = [(i,j) for i in range(g.rows) for j in range(g.cols) if g.visible[i][j] == Minesweeper.HIDDEN]
            if not unknowns:
                return {cell: 1.0 for cell in boundary}
            prob = (g.total_mines - g.flags) / len(unknowns)
            return {cell: prob for cell in boundary}
        probs = {boundary[i]: counts[i]/total_valid for i in range[n]}
        return probs

    def _fallback_pick(self):
        # pick an unknown cell with lowest estimated risk
        g = self.game
        unknowns = [(i,j) for i in range(g.rows) for j in range(g.cols) if g.visible[i][j] == Minesweeper.HIDDEN]
        if not unknowns: return None
        remaining_mines = g.total_mines - g.flags
        # simple global prob
        global_prob = remaining_mines / len(unknowns) if unknowns else 1.0
        # try to pick a cell not adjacent to revealed numbers first (safer heuristic)
        safe_candidates = [cell for cell in unknowns if all(g.visible[a][b] == Minesweeper.HIDDEN for a,b in g._neigh(*cell))]
        pick_list = safe_candidates if safe_candidates else unknowns
        return random.choice(pick_list)

# -------------------------
# Pygame UI
# -------------------------
CELL = 24
MARGIN = 5
TOP_MARGIN = 60

COLORS = {
    'bg': (25,25,25),
    'cell_hidden': (180,180,180),
    'cell_revealed': (220,220,220),
    'cell_flag': (255,200,0),
    'text': (0,0,0),
    'mine': (200,0,0),
    'grid': (150,150,150),
    'button': (70,70,70),
    'button_hover': (110,110,110),
    'button_text': (240,240,240)
}
NUM_COLORS = [(0,0,0),(0,0,255),(0,128,0),(255,0,0),(0,0,128),(128,0,0),(0,128,128),(0,0,0),(128,128,128)]

DIFFICULTIES = {
    "8x8":  (8, 8, 10),   # rows, cols, mines
    "16x16": (16, 16, 40)
}

def draw_board(screen, font, game):
    screen.fill(COLORS['bg'])
    rows, cols = game.rows, game.cols
    # draw info
    info = f"Mines: {game.total_mines}  Flags: {game.flags}  Revealed: {game.revealed_count}/{game.rows*game.cols - game.total_mines}"
    text = font.render(info, True, (240,240,240))
    screen.blit(text, (10,10))
    # draw cells
    for i in range(rows):
        for j in range(cols):
            x = MARGIN + j*(CELL+MARGIN)
            y = TOP_MARGIN + MARGIN + i*(CELL+MARGIN)
            rect = pygame.Rect(x,y,CELL,CELL)
            val = game.visible[i][j]
            if val == Minesweeper.HIDDEN:
                pygame.draw.rect(screen, COLORS['cell_hidden'], rect)
            elif val == Minesweeper.FLAG:
                pygame.draw.rect(screen, COLORS['cell_flag'], rect)
                f = font.render("F", True, COLORS['text'])
                screen.blit(f, (x+6,y+2))
            elif val == Minesweeper.MINE:
                pygame.draw.rect(screen, COLORS['mine'], rect)
            else:
                pygame.draw.rect(screen, COLORS['cell_revealed'], rect)
                if val > 0:
                    txt = font.render(str(val), True, NUM_COLORS[min(val,len(NUM_COLORS)-1)])
                    screen.blit(txt, (x+6,y+2))
            pygame.draw.rect(screen, COLORS['grid'], rect, 1)
    # game over message
    if game.game_over:
        s = "You win! (press click to restart)" if game.victory else "Game over (press click to restart)"
        msg = font.render(s, True, (255,255,255))
        screen.blit(msg, (10, 30))

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

def main():
    pygame.init()

    # We size the window for the largest board (16x16)
    max_rows, max_cols = 16, 16
    width = max_cols*CELL + (max_cols+1)*MARGIN
    height = max_rows*CELL + (max_rows+1)*MARGIN + TOP_MARGIN

    screen = pygame.display.set_mode((width, height))
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

    # Button layout for menu
    BUTTON_W, BUTTON_H = 200, 45
    start_rect = pygame.Rect(width//2 - BUTTON_W//2, 160, BUTTON_W, BUTTON_H)
    diff_rect = pygame.Rect(width//2 - BUTTON_W//2, 225, BUTTON_W, BUTTON_H)
    quit_rect = pygame.Rect(width//2 - BUTTON_W//2, 290, BUTTON_W, BUTTON_H)
    button_rects = (start_rect, diff_rect, quit_rect)

    running = True
    while running:
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
                    if game.game_over:
                        # restart current difficulty on click
                        game.reset()
                        bot = MinesweeperBot(game, enum_limit=20)
                    else:
                        mx,my = event.pos
                        if my > TOP_MARGIN:
                            j = (mx - MARGIN) // (CELL + MARGIN)
                            i = (my - TOP_MARGIN - MARGIN) // (CELL + MARGIN)
                            if 0 <= i < game.rows and 0 <= j < game.cols:
                                if event.button == 1:
                                    game.open(i,j)
                                elif event.button == 3:
                                    game.flag(i,j)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        auto_play = not auto_play
                    elif event.key == pygame.K_r:
                        game.reset()
                        bot = MinesweeperBot(game, enum_limit=20)
                    elif event.key == pygame.K_RIGHT:
                        # single bot step
                        action = bot.step()
                    elif event.key == pygame.K_UP:
                        bot.enum_limit += 1
                        print("enum_limit:", bot.enum_limit)
                    elif event.key == pygame.K_DOWN:
                        bot.enum_limit = max(5, bot.enum_limit-1)
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

            draw_board(screen, font, game)

        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    main()
