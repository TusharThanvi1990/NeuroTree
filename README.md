# Hybrid CNN + MCTS for Chess Position Evaluation

## Overview
This project implements a hybrid approach combining a Convolutional Neural Network (CNN) and Monte Carlo Tree Search (MCTS) to evaluate chess positions. The CNN model provides position evaluations, while MCTS is used for move selection based on these evaluations.

## Features
- **CNN Model:** Predicts evaluation scores for chess positions in the range [0,1].
- **Monte Carlo Tree Search (MCTS):** Uses the CNN evaluation function to guide the search for the best move.
- **Board Representation:** 8×8×19 input representation encoding piece placement, castling rights, turn, en passant squares, and move clock.
- **PGN Export:** Saves played games in Portable Game Notation (PGN) format.

## Model Architecture
The CNN model takes a chess position represented as an 8×8×19 tensor and outputs a single scalar evaluation value. The architecture includes:
- **Convolutional Layers:** Extract spatial features from the board.
- **Batch Normalization & Activation Layers:** Improve learning efficiency.
- **Fully Connected Layers:** Convert extracted features into a final evaluation.
- **Loss Function:** Binary Cross Entropy (can be modified if needed).

## Board Representation
The chess board is encoded as an 8×8×19 tensor using the `board_to_cnn_input` function:
```python
import numpy as np
import chess

def board_to_cnn_input(board):
    planes = np.zeros((8, 8, 19), dtype=np.float32)
    piece_map = board.piece_map()
    piece_to_plane = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
    }
    for square, piece in piece_map.items():
        rank, file = divmod(square, 8)
        idx = piece_to_plane[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
        planes[rank, file, idx] = 1
    planes[:, :, 12] = int(board.turn == chess.WHITE)
    planes[:, :, 13] = int(board.has_kingside_castling_rights(chess.WHITE))
    planes[:, :, 14] = int(board.has_queenside_castling_rights(chess.WHITE))
    planes[:, :, 15] = int(board.has_kingside_castling_rights(chess.BLACK))
    planes[:, :, 16] = int(board.has_queenside_castling_rights(chess.BLACK))
    if board.ep_square:
        ep_rank, ep_file = divmod(board.ep_square, 8)
        planes[ep_rank, ep_file, 17] = 1
    planes[:, :, 18] = board.halfmove_clock / 100
    return planes
```

## Using the Model
To predict evaluations for a given position:
```python
eval_score = eval_cnn_v2.predict(board_to_cnn_input(board))
print(f"Evaluation: {eval_score}")
```

## Monte Carlo Tree Search (MCTS)
MCTS uses the CNN model for position evaluation during search. The best move is selected based on multiple simulations:
```python
while not board.is_game_over():
    best_move = mcts.search(iterations=1000)
    board.push(best_move)
    print(board)
```

## Saving Games to PGN
Games can be saved in PGN format for analysis:
```python
import chess.pgn

game = chess.pgn.Game()
game.headers["Event"] = "CNN + MCTS Test"
n = game
board = chess.Board()

while not board.is_game_over():
    move = mcts.search(iterations=1000)
    board.push(move)
    n = n.add_variation(move)

with open("game.pgn", "w") as pgn_file:
    print(game, file=pgn_file)
```

## Installation
Ensure you have the following dependencies:
```bash
pip install numpy python-chess tensorflow
```

## Future Improvements
- Optimize CNN architecture for better accuracy.
- Implement Alpha-Beta pruning for faster move selection.
- Train on a larger dataset for better generalization.

## License
This project is open-source and free to use. Contributions are welcome!

