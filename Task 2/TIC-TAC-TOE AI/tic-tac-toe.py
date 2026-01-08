import math

# Initialize board
board = [" " for _ in range(9)]

# Print board
def print_board():
    for row in [board[i:i+3] for i in range(0, 9, 3)]:
        print("|".join(row))
        print("-" * 5)

# Check winner
def check_winner(player: str) -> bool:
    win_combos = [
        [0,1,2], [3,4,5], [6,7,8],
        [0,3,6], [1,4,7], [2,5,8],
        [0,4,8], [2,4,6]
    ]
    return any(all(board[i] == player for i in combo) for combo in win_combos)


def is_draw() -> bool:
    return " " not in board
def minimax(is_maximizing: bool) -> float:
    if check_winner("O"):
        return 1
    if check_winner("X"):
        return -1
    if is_draw():
        return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(9):
            if board[i] == " ":
                board[i] = "O"
                score = minimax(False)
                board[i] = " "
                best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for i in range(9):
            if board[i] == " ":
                board[i] = "X"
                score = minimax(True)
                board[i] = " "
                best_score = min(score, best_score)
        return best_score


def ai_move() -> None:
    best_score = -math.inf
    best_move: int | None = None

    for i in range(9):
        if board[i] == " ":
            board[i] = "O"
            score = minimax(False)
            board[i] = " "
            if score > best_score:
                best_score = score
                best_move = i

    if best_move is not None:
        board[best_move] = "O"

def human_move() -> None:
    while True:
        move = int(input("Enter your move (0-8): "))
        if board[move] == " ":
            board[move] = "X"
            break
        else:
            print("Invalid move. Try again.")

print("You are X | AI is O")
print_board()

while True:
    human_move()
    print_board()
    if check_winner("X"):
        print("🎉 You win!")
        break
    if is_draw():
        print("🤝 It's a draw!")
        break

    ai_move()
    print("AI played:")
    print_board()
    if check_winner("O"):
        print("🤖 AI wins!")
        break
    if is_draw():
        print("🤝 It's a draw!")
        break
