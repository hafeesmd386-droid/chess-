
import random
import sys

import pygame
import chess

# ============================================================
# CONFIGURATION
# ============================================================

pygame.init()

WIDTH, HEIGHT = 1000, 720
BOARD_SIZE = 640
SQ = BOARD_SIZE // 8
BOARD_X, BOARD_Y = 24, 40
PANEL_X = 700

SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Grandmaster Chess")

CLOCK = pygame.time.Clock()

# Theme
BG = (20, 25, 35)
PANEL = (29, 36, 49)
LIGHT = (235, 224, 204)
DARK = (104, 137, 126)
ACCENT = (95, 190, 155)
TEXT = (239, 243, 248)
MUTED = (160, 172, 190)
HIGHLIGHT = (246, 210, 92)
LAST_MOVE = (224, 190, 85)
LEGAL_DOT = (35, 45, 53)
WHITE_PIECE = (250, 250, 244)
BLACK_PIECE = (35, 40, 49)
RED = (235, 100, 100)

FONT = pygame.font.SysFont("arial", 20)
SMALL = pygame.font.SysFont("arial", 15)
TITLE = pygame.font.SysFont("arial", 29, bold=True)
PIECE_FONT = pygame.font.SysFont(
    "segoeuisymbol,dejavusans,arial", 62
)
BIG_PIECE_FONT = pygame.font.SysFont(
    "segoeuisymbol,dejavusans,arial", 76
)

PIECE_SYMBOLS = {
    (chess.WHITE, chess.KING): "\u2654",
    (chess.WHITE, chess.QUEEN): "\u2655",
    (chess.WHITE, chess.ROOK): "\u2656",
    (chess.WHITE, chess.BISHOP): "\u2657",
    (chess.WHITE, chess.KNIGHT): "\u2658",
    (chess.WHITE, chess.PAWN): "\u2659",
    (chess.BLACK, chess.KING): "\u265A",
    (chess.BLACK, chess.QUEEN): "\u265B",
    (chess.BLACK, chess.ROOK): "\u265C",
    (chess.BLACK, chess.BISHOP): "\u265D",
    (chess.BLACK, chess.KNIGHT): "\u265E",
    (chess.BLACK, chess.PAWN): "\u265F",
}

VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 335,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


# ============================================================
# CHESS ENGINE: MINIMAX WITH ALPHA-BETA PRUNING
# ============================================================

def evaluate(board):
    """Positive scores favor White; negative scores favor Black."""
    if board.is_checkmate():
        return -100000 if board.turn == chess.WHITE else 100000

    if board.is_game_over():
        return 0

    score = 0

    for square, piece in board.piece_map().items():
        value = VALUES[piece.piece_type]
        file_no = chess.square_file(square)
        rank_no = chess.square_rank(square)

        # Reward pieces for occupying the center.
        center_distance = abs(file_no - 3.5) + abs(rank_no - 3.5)
        center_bonus = int((7 - center_distance) * 4)

        if piece.piece_type == chess.PAWN:
            advance = rank_no if piece.color == chess.WHITE else 7 - rank_no
            center_bonus += advance * 5

        piece_score = value + center_bonus

        if piece.color == chess.WHITE:
            score += piece_score
        else:
            score -= piece_score

    # Keep the evaluation relative to the side to move.
    return score if board.turn == chess.WHITE else -score


def ordered_moves(board):
    """Try promising moves first to improve alpha-beta pruning."""
    moves = list(board.legal_moves)

    def move_score(move):
        score = 0

        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)

            victim_value = (
                VALUES[victim.piece_type] if victim else VALUES[chess.PAWN]
            )
            attacker_value = VALUES[attacker.piece_type] if attacker else 0
            score += 10 * victim_value - attacker_value

        if move.promotion:
            score += VALUES.get(move.promotion, 0) + 800

        return score

    moves.sort(key=move_score, reverse=True)
    return moves


def minimax(board, depth, alpha, beta):
    """Negamax search: the computer always maximizes its score."""
    if depth == 0 or board.is_game_over():
        return evaluate(board), None

    best_score = -float("inf")
    best_move = None

    for move in ordered_moves(board):
        board.push(move)
        score, _ = minimax(board, depth - 1, -beta, -alpha)
        score = -score
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, score)
        if alpha >= beta:
            break

    return best_score, best_move


def get_computer_move(board):
    """Search two plies ahead for a legal computer move."""
    _, move = minimax(board, 2, -float("inf"), float("inf"))
    return move


# ============================================================
# GAME STATE
# ============================================================

board = chess.Board()
selected_square = None
legal_targets = []
last_move = None
mode = "Computer"
human_color = chess.WHITE
message = "Your turn — White"
thinking = False
ai_timer = 0

# Buttons: (label, rectangle)
BUTTONS = {
    "new": ("New Game", pygame.Rect(PANEL_X, 430, 250, 44)),
    "undo": ("Undo Move", pygame.Rect(PANEL_X, 486, 250, 44)),
    "mode": ("Mode: Computer", pygame.Rect(PANEL_X, 542, 250, 44)),
}


# ============================================================
# DRAWING HELPERS
# ============================================================

def draw_text(text, x, y, font=FONT, color=TEXT):
    image = font.render(str(text), True, color)
    SCREEN.blit(image, (x, y))
    return image.get_rect(topleft=(x, y))


def draw_centered(text, rect, font=FONT, color=TEXT):
    image = font.render(text, True, color)
    SCREEN.blit(image, image.get_rect(center=rect.center))


def square_rect(square):
    file_no = chess.square_file(square)
    rank_no = chess.square_rank(square)

    x = BOARD_X + file_no * SQ
    y = BOARD_Y + (7 - rank_no) * SQ
    return pygame.Rect(x, y, SQ, SQ)


def draw_board():
    # Board background
    for rank_no in range(8):
        for file_no in range(8):
            square = chess.square(file_no, rank_no)
            rect = square_rect(square)

            color = LIGHT if (file_no + rank_no) % 2 == 0 else DARK
            pygame.draw.rect(SCREEN, color, rect)

            if last_move and square in (
                last_move.from_square,
                last_move.to_square,
            ):
                overlay = pygame.Surface((SQ, SQ), pygame.SRCALPHA)
                overlay.fill((*LAST_MOVE, 85))
                SCREEN.blit(overlay, rect.topleft)

            if selected_square == square:
                pygame.draw.rect(SCREEN, HIGHLIGHT, rect, 4)

            # File and rank coordinates
            if rank_no == 0:
                draw_text(
                    chess.FILE_NAMES[file_no],
                    rect.x + SQ - 15,
                    rect.y + SQ - 22,
                    SMALL,
                    DARK if color == LIGHT else LIGHT,
                )

            if file_no == 0:
                draw_text(
                    str(rank_no + 1),
                    rect.x + 5,
                    rect.y + 3,
                    SMALL,
                    DARK if color == LIGHT else LIGHT,
                )

    # Legal-move indicators
    for target in legal_targets:
        rect = square_rect(target)

        if board.piece_at(target):
            pygame.draw.circle(
                SCREEN, ACCENT, rect.center, SQ // 2 - 5, 4
            )
        else:
            pygame.draw.circle(
                SCREEN, LEGAL_DOT, rect.center, 9
            )

    # Chess pieces
    for square, piece in board.piece_map().items():
        rect = square_rect(square)
        symbol = PIECE_SYMBOLS[(piece.color, piece.piece_type)]

        # Subtle shadow improves contrast.
        shadow = PIECE_FONT.render(symbol, True, (45, 48, 48))
        piece_color = WHITE_PIECE if piece.color else BLACK_PIECE
        glyph = PIECE_FONT.render(symbol, True, piece_color)

        shadow_rect = shadow.get_rect(
            center=(rect.centerx + 2, rect.centery + 3)
        )
        glyph_rect = glyph.get_rect(center=rect.center)

        SCREEN.blit(shadow, shadow_rect)
        SCREEN.blit(glyph, glyph_rect)


def game_status():
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return f"CHECKMATE — {winner} wins!"

    if board.is_stalemate():
        return "Draw — stalemate"

    if board.is_insufficient_material():
        return "Draw — insufficient material"

    if board.is_seventyfive_moves():
        return "Draw — 75-move rule"

    if board.is_fivefold_repetition():
        return "Draw — fivefold repetition"

    if board.is_check():
        side = "White" if board.turn == chess.WHITE else "Black"
        return f"{side} is in check!"

    if board.is_game_over(claim_draw=True):
        return "Draw"

    side = "White" if board.turn == chess.WHITE else "Black"
    if mode == "Computer" and board.turn != human_color:
        return "Computer is thinking..."

    return f"{side} to move"


def draw_panel():
    panel_rect = pygame.Rect(680, 0, WIDTH - 680, HEIGHT)
    pygame.draw.rect(SCREEN, PANEL, panel_rect)

    draw_text("GRANDMASTER", PANEL_X, 32, TITLE, ACCENT)
    draw_text("CHESS", PANEL_X, 66, TITLE, TEXT)

    pygame.draw.line(
        SCREEN, (65, 76, 92), (PANEL_X, 112), (960, 112), 2
    )

    draw_text("GAME STATUS", PANEL_X, 135, SMALL, MUTED)

    status = game_status()
    status_color = RED if board.is_check() or board.is_checkmate() else ACCENT

    # Wrap longer status messages.
    words = status.split()
    lines = []
    line = ""

    for word in words:
        candidate = (line + " " + word).strip()
        if FONT.size(candidate)[0] > 255:
            lines.append(line)
            line = word
        else:
            line = candidate

    if line:
        lines.append(line)

    for i, line in enumerate(lines):
        draw_text(line, PANEL_X, 163 + i * 28, FONT, status_color)

    # Turn indicator
    side = "White" if board.turn == chess.WHITE else "Black"
    draw_text("SIDE TO MOVE", PANEL_X, 230, SMALL, MUTED)
    draw_text(side, PANEL_X, 253, FONT, TEXT)

    if board.is_check():
        draw_text("CHECK!", PANEL_X + 105, 253, FONT, RED)

    pygame.draw.line(
        SCREEN, (65, 76, 92), (PANEL_X, 300), (960, 300), 2
    )

    draw_text("MOVE HISTORY", PANEL_X, 318, SMALL, MUTED)

    # Show recent moves in algebraic notation.
    history = []
    temp_board = chess.Board()

    for move in board.move_stack:
        san = temp_board.san(move)
        history.append(san)
        temp_board.push(move)

    start = max(0, len(history) - 8)
    recent = history[start:]

    for i in range(0, len(recent), 2):
        move_number = start // 2 + i // 2 + 1
        white_move = recent[i]
        black_move = recent[i + 1] if i + 1 < len(recent) else ""

        draw_text(
            f"{move_number}. {white_move}   {black_move}",
            PANEL_X,
            348 + (i // 2) * 23,
            SMALL,
            TEXT,
        )

    for key, (label, rect) in BUTTONS.items():
        color = ACCENT if key == "new" else (54, 65, 81)
        pygame.draw.rect(SCREEN, color, rect, border_radius=9)
        draw_centered(label, rect, FONT, BG if key == "new" else TEXT)

    draw_text("Click a piece, then a square.", PANEL_X, 610, SMALL, MUTED)
    draw_text("Choose Undo to take back a move.", PANEL_X, 634, SMALL, MUTED)
    draw_text("Press ESC to quit.", PANEL_X, 658, SMALL, MUTED)


def draw_screen():
    SCREEN.fill(BG)
    draw_board()
    draw_panel()
    pygame.display.flip()


# ============================================================
# GAME ACTIONS
# ============================================================

def reset_game():
    global board, selected_square, legal_targets
    global last_move, message, thinking

    board = chess.Board()
    selected_square = None
    legal_targets = []
    last_move = None
    message = "Your turn — White"
    thinking = False


def undo_move():
    global selected_square, legal_targets, last_move, thinking

    if not board.move_stack:
        return

    # In computer mode, undo both the computer's move and your move
    # when possible, returning the game to the human's turn.
    board.pop()

    if mode == "Computer" and board.move_stack:
        board.pop()

    selected_square = None
    legal_targets = []
    last_move = board.peek() if board.move_stack else None
    thinking = False


def handle_board_click(pos):
    global selected_square, legal_targets, last_move

    x, y = pos

    if not (
        BOARD_X <= x < BOARD_X + BOARD_SIZE
        and BOARD_Y <= y < BOARD_Y + BOARD_SIZE
    ):
        return

    file_no = (x - BOARD_X) // SQ
    rank_no = 7 - ((y - BOARD_Y) // SQ)
    square = chess.square(file_no, rank_no)

    # Only accept human input during the human's turn.
    if mode == "Computer" and board.turn != human_color:
        return

    if board.is_game_over(claim_draw=True):
        return

    if selected_square is not None:
        candidates = [
            move for move in board.legal_moves
            if move.from_square == selected_square
            and move.to_square == square
        ]

        if candidates:
            move = candidates[0]

            # Promote automatically to a queen.
            if move.promotion:
                queen_move = chess.Move(
                    move.from_square,
                    move.to_square,
                    promotion=chess.QUEEN,
                )
                if queen_move in board.legal_moves:
                    move = queen_move

            board.push(move)
            last_move = move
            selected_square = None
            legal_targets = []
            return

    piece = board.piece_at(square)

    if piece and piece.color == board.turn:
        selected_square = square
        legal_targets = [
            move.to_square
            for move in board.legal_moves
            if move.from_square == square
        ]
    else:
        selected_square = None
        legal_targets = []


def handle_button_click(pos):
    global mode, human_color, selected_square, legal_targets

    for key, (label, rect) in BUTTONS.items():
        if not rect.collidepoint(pos):
            continue

        if key == "new":
            reset_game()

        elif key == "undo":
            undo_move()

        elif key == "mode":
            mode = "Two Player" if mode == "Computer" else "Computer"
            BUTTONS["mode"] = (
                f"Mode: {mode}",
                BUTTONS["mode"][1],
            )
            reset_game()

        selected_square = None
        legal_targets = []
        return


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    global thinking, last_move

    running = True

    while running:
        CLOCK.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_n:
                    reset_game()

                elif event.key == pygame.K_u:
                    undo_move()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if event.pos[0] >= 680:
                    handle_button_click(event.pos)
                else:
                    handle_board_click(event.pos)

        # Computer plays Black. Search is deliberately kept simple
        # so the game runs without downloading an external engine.
        if (
            mode == "Computer"
            and board.turn != human_color
            and not board.is_game_over(claim_draw=True)
        ):
            thinking = True
            draw_screen()

            move = get_computer_move(board)

            if move is not None:
                board.push(move)
                last_move = move

            thinking = False

        draw_screen()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
