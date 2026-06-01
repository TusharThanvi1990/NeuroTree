"""
AlphaZero-style MCTS using the dual-head chess model (policy + value).

Usage:
    import chess
    from tensorflow.keras.models import load_model
    from mcts import AlphaZeroMCTS

    model = load_model("chess_model.keras")
    mcts = AlphaZeroMCTS(model)
    board = chess.Board()
    move = mcts.search(board, num_simulations=400)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import chess
import numpy as np

Move = chess.Move


def board_to_cnn_input(board: chess.Board) -> np.ndarray:
    planes = np.zeros((8, 8, 19), dtype=np.float32)
    piece_to_plane = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        idx = piece_to_plane[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
        planes[rank, file, idx] = 1.0

    planes[:, :, 12] = float(board.turn == chess.WHITE)
    planes[:, :, 13] = float(board.has_kingside_castling_rights(chess.WHITE))
    planes[:, :, 14] = float(board.has_queenside_castling_rights(chess.WHITE))
    planes[:, :, 15] = float(board.has_kingside_castling_rights(chess.BLACK))
    planes[:, :, 16] = float(board.has_queenside_castling_rights(chess.BLACK))

    if board.ep_square is not None:
        ep_rank, ep_file = divmod(board.ep_square, 8)
        planes[ep_rank, ep_file, 17] = 1.0

    planes[:, :, 18] = board.halfmove_clock / 100.0
    return planes


def generate_all_uci_labels() -> List[str]:
    """UCI labels aligned with the dataset generation script in Google-Colab-Script.ipynb."""
    labels: List[str] = []
    for from_square in range(64):
        for to_square in range(64):
            if from_square == to_square:
                continue
            uci = chess.SQUARE_NAMES[from_square] + chess.SQUARE_NAMES[to_square]
            labels.append(uci)
            if chess.square_rank(from_square) in (1, 6):
                for promo in ("q", "r", "b", "n"):
                    labels.append(uci + promo)
    labels.append("None")
    return labels


def build_move_index(policy_size: int) -> Dict[str, int]:
    labels = generate_all_uci_labels()
    if len(labels) < policy_size:
        raise ValueError(
            f"Policy size {policy_size} exceeds generated label count {len(labels)}"
        )
    return {uci: idx for idx, uci in enumerate(labels[:policy_size])}


def terminal_value(board: chess.Board) -> float:
    """Value in [-1, 1] from the perspective of the side to move."""
    if board.is_checkmate():
        return -1.0
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0.0
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return 0.0
    if outcome.winner == board.turn:
        return 1.0
    return -1.0


def normalized_eval_to_value(eval_score: float, turn: chess.Color) -> float:
    """Map training target in [0, 1] (white advantage) to [-1, 1] for side to move."""
    white_adv = 2.0 * float(eval_score) - 1.0
    return white_adv if turn == chess.WHITE else -white_adv


class MCTSNode:
    __slots__ = ("board", "parent", "move", "prior", "children", "visit_count", "value_sum")

    def __init__(
        self,
        board: chess.Board,
        parent: Optional[MCTSNode] = None,
        move: Optional[Move] = None,
        prior: float = 0.0,
    ) -> None:
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children: Dict[Move, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0


class AlphaZeroMCTS:
    """PUCT MCTS with policy priors and value head; root move = max visit count."""

    def __init__(
        self,
        model,
        cpuct: float = 1.5,
        move_index: Optional[Dict[str, int]] = None,
    ) -> None:
        self.model = model
        self.cpuct = cpuct
        self.policy_size = self._infer_policy_size(model)
        self.move_index = move_index or build_move_index(self.policy_size)
        self._has_policy_head = self._detect_policy_head(model)

    @staticmethod
    def _infer_policy_size(model) -> int:
        if hasattr(model, "get_layer"):
            try:
                layer = model.get_layer("move_output")
                return int(layer.output_shape[-1])
            except ValueError:
                pass
        outputs = model.outputs if hasattr(model, "outputs") else None
        if outputs and len(outputs) >= 1:
            shape = outputs[0].shape
            if shape[-1] is not None:
                return int(shape[-1])
        raise ValueError("Could not infer policy output size from model")

    @staticmethod
    def _detect_policy_head(model) -> bool:
        if hasattr(model, "output_names") and model.output_names:
            return "move_output" in model.output_names
        return len(getattr(model, "outputs", []) or []) >= 2

    def _predict(self, board: chess.Board) -> Tuple[Dict[Move, float], float]:
        x = np.expand_dims(board_to_cnn_input(board), axis=0)
        raw = self.model.predict(x, verbose=0)

        if isinstance(raw, dict):
            policy_logits = np.asarray(raw["move_output"][0], dtype=np.float64)
            eval_score = float(np.asarray(raw["eval_output"]).reshape(-1)[0])
        elif isinstance(raw, (list, tuple)):
            if len(raw) >= 2:
                policy_logits = np.asarray(raw[0][0], dtype=np.float64)
                eval_score = float(np.asarray(raw[1]).reshape(-1)[0])
            else:
                policy_logits = None
                eval_score = float(np.asarray(raw[0]).reshape(-1)[0])
        else:
            policy_logits = None
            eval_score = float(np.asarray(raw).reshape(-1)[0])

        value = normalized_eval_to_value(eval_score, board.turn)
        priors = self._policy_for_legal_moves(board, policy_logits)
        return priors, value

    def _policy_for_legal_moves(
        self, board: chess.Board, policy_logits: Optional[np.ndarray]
    ) -> Dict[Move, float]:
        legal = list(board.legal_moves)
        if not legal:
            return {}

        if policy_logits is None or not self._has_policy_head:
            p = 1.0 / len(legal)
            return {m: p for m in legal}

        indices: List[int] = []
        mapped_moves: List[Move] = []
        for move in legal:
            uci = move.uci()
            idx = self.move_index.get(uci)
            if idx is None:
                continue
            indices.append(idx)
            mapped_moves.append(move)

        if not mapped_moves:
            p = 1.0 / len(legal)
            return {m: p for m in legal}

        logits = policy_logits[indices]
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        probs = exp / np.sum(exp)

        priors = {m: float(p) for m, p in zip(mapped_moves, probs)}
        missing = [m for m in legal if m not in priors]
        if missing:
            share = sum(priors.values())
            leftover = max(0.0, 1.0 - share)
            extra = leftover / len(missing) if missing else 0.0
            for m in missing:
                priors[m] = extra
            total = sum(priors.values())
            if total > 0:
                priors = {m: p / total for m, p in priors.items()}
        return priors

    def _expand(self, node: MCTSNode) -> float:
        board = node.board
        if board.is_game_over():
            return terminal_value(board)

        if node.children:
            _, value = self._predict(board)
            return value

        priors, value = self._predict(board)
        for move, prior in priors.items():
            child_board = board.copy()
            child_board.push(move)
            node.children[move] = MCTSNode(child_board, parent=node, move=move, prior=prior)
        return value

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        total_visits = sum(child.visit_count for child in node.children.values())
        sqrt_total = math.sqrt(total_visits + 1)

        best_score = -float("inf")
        best: Optional[MCTSNode] = None
        for child in node.children.values():
            q = -child.value_sum / (child.visit_count + 1e-8)
            u = self.cpuct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best = child
        assert best is not None
        return best

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        current: Optional[MCTSNode] = node
        v = value
        while current is not None:
            current.visit_count += 1
            current.value_sum += v
            v = -v
            current = current.parent

    def search(self, board: chess.Board, num_simulations: int = 400) -> Optional[Move]:
        root_board = board.copy()
        root = MCTSNode(root_board)

        if root_board.is_game_over():
            return None

        self._expand(root)

        for _ in range(num_simulations):
            node = root
            while node.children and not node.board.is_game_over():
                node = self._select_child(node)

            if node.board.is_game_over():
                leaf_value = terminal_value(node.board)
            else:
                leaf_value = self._expand(node)

            self._backpropagate(node, leaf_value)

        return self.best_move(root)

    def best_move(self, root: MCTSNode) -> Optional[Move]:
        """AlphaZero: play the move with the highest visit count at the root."""
        if not root.children:
            return None
        return max(root.children.items(), key=lambda item: item[1].visit_count)[0]

    def root_policy(self, root: MCTSNode) -> Dict[Move, float]:
        """Visit distribution at the root (useful for training targets)."""
        total = sum(child.visit_count for child in root.children.values())
        if total == 0:
            return {}
        return {move: child.visit_count / total for move, child in root.children.items()}


# Backward-compatible alias for notebooks that used class name MCTS
MCTS = AlphaZeroMCTS
