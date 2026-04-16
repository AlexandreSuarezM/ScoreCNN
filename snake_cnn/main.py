"""
main.py – Pygame visualisation for Snake CNN agent.

Controls:
  +  /  =   speed up  (more game steps / second)
  -         slow down
  R         restart game
  ESC       quit

Layout:
  Left  : 20×20 snake grid  (560×560 px)
  Right : info panel showing CNN + heuristic probabilities and architecture

The game loop runs at 60 FPS (smooth render). Game logic advances at
`speed` steps-per-second – fully synchronous with the CNN; no frames
are skipped. The CNN forward pass completes before the next step.
"""
import sys
import time
import pygame
import numpy as np

from snake_game import SnakeGame
from cnn        import HomemadeCNN, ARCH_1CONV, ARCH_DEEP, DEFAULT_ARCH
from agent      import SnakeAgent

# ── layout ────────────────────────────────────────
GRID      = 20
CELL      = 28                    # pixels per grid cell
PANEL_W   = 280
GW        = GRID * CELL           # grid pixel width  (560)
GH        = GRID * CELL           # grid pixel height (560)
WIN_W     = GW + PANEL_W
WIN_H     = GH

# ── colours ───────────────────────────────────────
C_BG        = (15,  15,  25)
C_GRID      = (32,  32,  48)
C_HEAD      = (110, 255, 110)
C_BODY_MAX  = 200
C_FRUIT     = (220, 55,  55)
C_PANEL     = (18,  18,  32)
C_SEP       = (50,  50,  75)
C_WHITE     = (225, 225, 225)
C_GRAY      = (130, 130, 155)
C_YELLOW    = (255, 210, 0)
C_CYAN      = (70,  195, 255)
C_ORANGE    = (255, 155, 50)
C_GREEN     = (60,  200, 60)
C_RED       = (215, 50,  50)
C_DARKBAR   = (45,  45,  60)

RESTART_DELAY = 1.5               # seconds before auto-restart after death


# ── drawing helpers ───────────────────────────────

def bar(surf, x, y, w, h, fill: float, color):
    """Horizontal progress bar [0, 1]."""
    pygame.draw.rect(surf, C_DARKBAR, (x, y, w, h), border_radius=2)
    fw = max(2, int(fill * w))
    pygame.draw.rect(surf, color, (x, y, fw, h), border_radius=2)


def draw_grid(surf, game: SnakeGame):
    surf.fill(C_BG, (0, 0, GW, GH))

    # grid lines
    for i in range(GRID + 1):
        pygame.draw.line(surf, C_GRID, (i * CELL, 0),  (i * CELL, GH))
        pygame.draw.line(surf, C_GRID, (0, i * CELL),  (GW, i * CELL))

    # snake body – green gradient, brighter toward head
    snake_list = list(game.snake)
    for idx, (r, c) in enumerate(snake_list[1:], 1):
        intensity = max(35, C_BODY_MAX - idx * 5)
        col = (0, intensity, 0)
        pygame.draw.rect(surf, col,
                         (c * CELL + 1, r * CELL + 1, CELL - 2, CELL - 2),
                         border_radius=3)

    # head
    hr, hc = snake_list[0]
    pygame.draw.rect(surf, C_HEAD,
                     (hc * CELL + 1, hr * CELL + 1, CELL - 2, CELL - 2),
                     border_radius=4)

    # fruit (circle)
    if game.fruit:
        fr, fc = game.fruit
        pygame.draw.ellipse(surf, C_FRUIT,
                            (fc * CELL + 4, fr * CELL + 4, CELL - 8, CELL - 8))

    # red overlay on death
    if game.done:
        ov = pygame.Surface((GW, GH), pygame.SRCALPHA)
        ov.fill((200, 0, 0, 55))
        surf.blit(ov, (0, 0))


def draw_panel(surf, game: SnakeGame,
               final_probs: np.ndarray, cnn_probs: np.ndarray,
               speed: int, arch_lines: list, fonts):
    big, med, sm = fonts
    px = GW + 12
    py = 10
    bw = PANEL_W - 28           # bar width

    surf.fill(C_PANEL, (GW, 0, PANEL_W, WIN_H))
    pygame.draw.line(surf, C_SEP, (GW, 0), (GW, WIN_H), 2)

    # ── title ────────────────────────────────────
    surf.blit(big.render("SNAKE CNN", True, C_WHITE), (px, py)); py += 32

    # ── stats ────────────────────────────────────
    surf.blit(med.render(f"Score : {game.score}", True, C_YELLOW), (px, py)); py += 22
    surf.blit(sm .render(f"Steps : {game.steps}", True, C_GRAY),   (px, py)); py += 17
    surf.blit(sm .render(f"Speed : {speed} sps   (+/- keys)",
                         True, C_GRAY), (px, py)); py += 22

    # ── CNN + Score combined ──────────────────────
    surf.blit(med.render("CNN + Score", True, C_CYAN), (px, py)); py += 20
    best = int(np.argmax(final_probs))
    for i, name in enumerate(("UP", "DOWN", "LEFT", "RIGHT")):
        col = C_GREEN if i == best else (75, 75, 110)
        bar(surf, px, py, bw, 16, float(final_probs[i]), col)
        surf.blit(sm.render(f"{name:5s} {final_probs[i]:.3f}", True, C_WHITE),
                  (px + 3, py)); py += 20
    py += 6

    # ── Raw CNN logits ────────────────────────────
    surf.blit(med.render("Raw CNN", True, C_ORANGE), (px, py)); py += 20
    best_c = int(np.argmax(cnn_probs))
    for i, name in enumerate(("UP", "DOWN", "LEFT", "RIGHT")):
        col = (200, 105, 40) if i == best_c else (75, 75, 110)
        bar(surf, px, py, bw, 16, float(cnn_probs[i]), col)
        surf.blit(sm.render(f"{name:5s} {cnn_probs[i]:.3f}", True, C_WHITE),
                  (px + 3, py)); py += 20
    py += 10

    # ── architecture ─────────────────────────────
    surf.blit(med.render("Architecture", True, (155, 215, 155)), (px, py)); py += 20
    surf.blit(sm.render("Input  20x20x3", True, C_GRAY), (px, py)); py += 15
    for line in arch_lines:
        surf.blit(sm.render(line, True, C_GRAY), (px, py)); py += 15
    surf.blit(sm.render("-> Softmax -> action", True, C_GRAY), (px, py)); py += 20

    # ── status ───────────────────────────────────
    if game.done:
        surf.blit(med.render("DEAD – restarting…", True, C_RED), (px, py))
    else:
        surf.blit(sm.render("R = restart  ESC = quit", True, C_GRAY), (px, py))


# ── main loop ─────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Snake AI – Homemade CNN")
    clock = pygame.time.Clock()

    fonts = (
        pygame.font.SysFont("monospace", 20, bold=True),   # big
        pygame.font.SysFont("monospace", 15, bold=True),   # med
        pygame.font.SysFont("monospace", 12),              # small
    )

    # ── choose architecture ──────────────────────
    # Switch arch= to experiment:
    #   DEFAULT_ARCH  → 2 conv + 2 dense  (recommended)
    #   ARCH_1CONV    → 1 conv + 1 dense  (minimal)
    #   ARCH_DEEP     → 4 layers with pooling (deeper)
    arch  = DEFAULT_ARCH
    game  = SnakeGame(GRID)
    agent = SnakeAgent(arch=arch)

    # Load best trained weights if available
    import os
    weights_path = "results/score_attract_weights.npy"
    if os.path.exists(weights_path):
        agent.cnn.set_weights(np.load(weights_path))
        print(f"Loaded weights: {weights_path}")

    arch_lines = agent.cnn.layer_summary()

    speed      = 8                      # game steps per second
    step_dt    = 1.0 / speed
    last_step  = time.perf_counter()
    death_time = None

    final_probs = np.full(4, 0.25, dtype=np.float32)
    cnn_probs   = np.full(4, 0.25, dtype=np.float32)

    running = True
    while running:

        # ── events ───────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    speed    = min(30, speed + 1)
                    step_dt  = 1.0 / speed
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed    = max(1, speed - 1)
                    step_dt  = 1.0 / speed
                elif ev.key == pygame.K_r:
                    game.reset()
                    death_time = None

        now = time.perf_counter()

        # ── auto-restart after death ─────────────
        if game.done:
            if death_time is None:
                death_time = now
            elif now - death_time > RESTART_DELAY:
                game.reset()
                death_time = None

        # ── game step (CNN + move) ────────────────
        elif now - last_step >= step_dt:
            # CNN runs here – synchronous, no skipping
            action, final_probs, cnn_probs = agent.decide(game)
            game.step(action)
            last_step = now

        # ── render ───────────────────────────────
        draw_grid(screen, game)
        draw_panel(screen, game, final_probs, cnn_probs,
                   speed, arch_lines, fonts)
        pygame.display.flip()
        clock.tick(60)              # display at 60 FPS regardless of game speed

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
