"""
Complete the methods below to implement your AI player.
You may add helper methods and classes as needed.

DO NOT modify the method signatures of __init__ or get_move.
"""

import time
from typing import Tuple, List, Dict

class Player:
    def __init__(self, player_number: int):
        """
        
        Args:
            player_number: 1 (White, starts at bottom) or 2 (Black, starts at top)
        """
        self.player_number = player_number
        self.opponent_number = 3 - player_number
        self.time_limit = 0
        self.start_time = 0
        self.nodes_searched = 0
        self.max_depth_reached = 0
        
    def get_move(self, state: Dict, time_limit: float) -> Tuple[int, int, int, int]:
        """
        Args:
            state: Dictionary containing game state
            time_limit: Maximum time allowed in seconds
        
        Returns:
            Tuple (from_row, from_col, to_row, to_col) representing your move
        """
        self.time_limit = time_limit * 0.60  
        self.start_time = time.time()
        self.nodes_searched = 0
        self.max_depth_reached = 0
        
        board = state['board']
        legal_moves = self.get_legal_moves(board, self.player_number)
        
        if not legal_moves:
            raise ValueError("No legal moves available!")
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Check for immediate winning moves
        for move in legal_moves:
            new_board = self.apply_move(board, move, self.player_number)
            if self.is_winning_position(new_board, self.player_number):
                return move
        
        # Iterative deepening
        best_move = legal_moves[0]
        best_score = float('-inf')
        
        for depth in range(1, 30):
            # Stop if we don't have enough time for next depth
            elapsed = time.time() - self.start_time
            if elapsed > self.time_limit * 0.5:  # Used more than 50%
                break
            
            try:
                move, score = self.search_depth(board, depth, legal_moves)
                
                if move is not None:
                    best_move = move
                    best_score = score
                    self.max_depth_reached = depth
                    
                    # Found a winning move
                    if score > 500000:
                        break
                        
            except TimeoutError:
                break
        
        return best_move
    
    def time_remaining(self):
        """Check how much time is left"""
        return self.time_limit - (time.time() - self.start_time)
    
    def search_depth(self, board, max_depth, legal_moves):
        """
        Search to a specific depth with move ordering.
        """
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        # Order moves for better pruning
        ordered_moves = self.order_moves(board, legal_moves, self.player_number)
        
        for move in ordered_moves:
            # Check time BEFORE processing each move
            if self.time_remaining() < 0.05:
                raise TimeoutError()
            
            new_board = self.apply_move(board, move, self.player_number)
            
            # Check for immediate win
            if self.is_winning_position(new_board, self.player_number):
                return move, 1000000
            
            # Minimax search
            score = self.minimax(new_board, max_depth - 1, alpha, beta, False, 1)
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
        
        return best_move, best_score
    
    def minimax(self, board, depth, alpha, beta, is_maximizing, ply):
        """
        Minimax with alpha-beta pruning and proper time checks.
        """
        # Check time at EVERY level
        if self.time_remaining() < 0.02:
            raise TimeoutError()
        
        self.nodes_searched += 1
        
        # Terminal depth
        if depth == 0:
            return self.evaluate_board(board)
        
        # Check for game over
        if self.is_winning_position(board, self.player_number):
            return 1000000 + depth
        if self.is_winning_position(board, self.opponent_number):
            return -1000000 - depth
        
        current_player = self.player_number if is_maximizing else self.opponent_number
        legal_moves = self.get_legal_moves(board, current_player)
        
        if not legal_moves:
            return 1000000 + depth if not is_maximizing else -1000000 - depth
        
        # Move ordering for efficiency
        ordered_moves = self.order_moves(board, legal_moves, current_player)
        
        if is_maximizing:
            max_eval = float('-inf')
            for move in ordered_moves:
                new_board = self.apply_move(board, move, current_player)
                eval_score = self.minimax(new_board, depth - 1, alpha, beta, False, ply + 1)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            for move in ordered_moves:
                new_board = self.apply_move(board, move, current_player)
                eval_score = self.minimax(new_board, depth - 1, alpha, beta, True, ply + 1)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval
    
    def order_moves(self, board, moves, player):
        """
        Order moves for maximum alpha-beta efficiency.
        Priority: winning moves > captures toward goal > advancement > center
        """
        scored_moves = []
        
        for move in moves:
            from_row, from_col, to_row, to_col = move
            score = 0
            
            # 1. Check if this move wins immediately
            new_board = self.apply_move(board, move, player)
            if self.is_winning_position(new_board, player):
                return [move]  # Only search this move
            
            # 2. Captures (highest priority after wins)
            if board[to_row][to_col] != 0:
                score += 8000
                # Captures near enemy goal are better
                if player == 1:
                    score += to_row * 100
                else:
                    score += (7 - to_row) * 100
            
            # 3. Advancement (exponential reward)
            if player == 1:
                score += (to_row ** 2) * 5
                # Major bonus for pieces near goal
                if to_row >= 6:
                    score += 2000
                if to_row == 7:
                    score += 10000
            else:
                score += ((7 - to_row) ** 2) * 5
                if to_row <= 1:
                    score += 2000
                if to_row == 0:
                    score += 10000
            
            # 4. Center control (columns 2-5)
            if 2 <= to_col <= 5:
                score += 30
            
            # 5. Prefer forward moves over diagonal
            if from_col == to_col:
                score += 10
            
            scored_moves.append((move, -score))
        
        scored_moves.sort(key=lambda x: x[1])
        return [move for move, _ in scored_moves]
    
    def evaluate_board(self, board):
        score = 0
        
        my_pieces = []
        opp_pieces = []
        
        # Single pass collection
        for row in range(8):
            for col in range(8):
                if board[row][col] == self.player_number:
                    my_pieces.append((row, col))
                elif board[row][col] == self.opponent_number:
                    opp_pieces.append((row, col))
        
        # 1. Material advantage (most important in endgame)
        material = (len(my_pieces) - len(opp_pieces)) * 200
        score += material
        
        # Early return for huge material advantages
        if len(my_pieces) - len(opp_pieces) >= 5:
            return 20000
        if len(opp_pieces) - len(my_pieces) >= 5:
            return -20000
        
        # 2. Piece advancement (primary strategy)
        my_advancement = 0
        opp_advancement = 0
        
        for row, col in my_pieces:
            if self.player_number == 1:
                progress = row
                my_advancement += progress ** 2
                # Huge bonus for pieces close to winning
                if row >= 6:
                    score += 400
                if row == 7:
                    return 1000000
            else:
                progress = 7 - row
                my_advancement += progress ** 2
                if row <= 1:
                    score += 400
                if row == 0:
                    return 1000000
        
        for row, col in opp_pieces:
            if self.opponent_number == 1:
                progress = row
                opp_advancement += progress ** 2
                if row >= 6:
                    score -= 400
                if row == 7:
                    return -1000000
            else:
                progress = 7 - row
                opp_advancement += progress ** 2
                if row <= 1:
                    score -= 400
                if row == 0:
                    return -1000000
        
        score += (my_advancement - opp_advancement) * 5
        
        # 3. Tactical threats (pieces that can be captured)
        my_threats = 0
        opp_threats = 0
        
        direction = 1 if self.player_number == 1 else -1
        opp_direction = -direction
        
        # Count pieces under attack
        for row, col in my_pieces:
            threat_row = row + direction
            if 0 <= threat_row < 8:
                if (col > 0 and board[threat_row][col - 1] == self.opponent_number) or \
                   (col < 7 and board[threat_row][col + 1] == self.opponent_number):
                    # Check if protected
                    protect_row = row - direction
                    protected = False
                    if 0 <= protect_row < 8:
                        if (col > 0 and board[protect_row][col - 1] == self.player_number) or \
                           (col < 7 and board[protect_row][col + 1] == self.player_number):
                            protected = True
                    
                    if not protected:
                        score -= 50  # Unprotected piece under attack
        
        # Count opponent pieces we threaten
        for row, col in opp_pieces:
            threat_row = row + opp_direction
            if 0 <= threat_row < 8:
                if (col > 0 and board[threat_row][col - 1] == self.player_number) or \
                   (col < 7 and board[threat_row][col + 1] == self.player_number):
                    opp_threats += 1
        
        score += opp_threats * 30
        
        # 4. Center control bonus
        my_center = sum(1 for r, c in my_pieces if 2 <= c <= 5)
        opp_center = sum(1 for r, c in opp_pieces if 2 <= c <= 5)
        score += (my_center - opp_center) * 15
        
        return score
    
    def is_winning_position(self, board, player):
        """Check if a player has won"""
        if player == 1:
            return any(board[7][col] == 1 for col in range(8))
        else:
            return any(board[0][col] == 2 for col in range(8))
    
    def apply_move(self, board, move, player):
        """Apply a move and return new board (does not modify original)"""
        new_board = [row[:] for row in board]
        from_row, from_col, to_row, to_col = move
        new_board[to_row][to_col] = player
        new_board[from_row][from_col] = 0
        return new_board
    
    def get_legal_moves(self, board: List[List[int]], player: int) -> List[Tuple[int, int, int, int]]:
        """Get all legal moves for a player."""
        moves = []
        direction = 1 if player == 1 else -1
        
        for row in range(8):
            for col in range(8):
                if board[row][col] == player:
                    new_row = row + direction
                    
                    if 0 <= new_row < 8:
                        # Straight forward (only if empty)
                        if board[new_row][col] == 0:
                            moves.append((row, col, new_row, col))
                        
                        # Diagonal left
                        if col > 0 and board[new_row][col - 1] != player:
                            moves.append((row, col, new_row, col - 1))
                        
                        # Diagonal right
                        if col < 7 and board[new_row][col + 1] != player:
                            moves.append((row, col, new_row, col + 1))
        
        return moves