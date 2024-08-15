"""
adapted and simplified from https://github.com/likenneth/othello_world/blob/master/data/othello.py
"""

import os

import numpy as np
import random
import torch

rows = list("abcdefgh")
columns = [str(i) for i in range(1, 9)]
eights = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]

def move_to_str(move):
    r, c = move // 8, move % 8
    return "".join([rows[r], columns[c]])

def generate_game(_):
    tbr = []
    game = OthelloGame()

    possible_next_steps = game.get_valid_moves()
    while possible_next_steps:
        next_step = random.choice(possible_next_steps)
        tbr.append(next_step)
        game.update([next_step,])
        possible_next_steps = game.get_valid_moves()

    return tbr

class OthelloGame():
    def __init__(self, board_size = 8):
        self.board_size = board_size * board_size

        board = np.zeros((8, 8))
        board[3, 4] = 1
        board[3, 3] = -1
        board[4, 3] = 1
        board[4, 4] = -1
        self.initial_state = board
        self.state = self.initial_state

        self.next_hand_color = 1 # 1 is black, -1 is white
        self.history = []

    @staticmethod
    def get_tbf(state, color, move):
        """
        given a game state, color and a move (`color` plays `move`),
        returns tbf, a list containing all 1-color pieces to be flipped
        """

        r, c = move // 8, move % 8
        tbf = []

        for direction in eights:
            buffer = []
            cur_r, cur_c = r, c
            while 1:
                cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                if cur_r < 0  or cur_r > 7 or cur_c < 0 or cur_c > 7:
                    break
                if state[cur_r, cur_c] == 0:
                    break
                elif state[cur_r, cur_c] == color:
                    tbf.extend(buffer)
                    break
                else:
                    buffer.append([cur_r, cur_c])
        return tbf
    
    def update(self, moves, prt=False):
        """
        takes a new move or new moves and update state
        """

        if prt:
            self.__print__()
        for move in moves:
            self.play_move(move)
            if prt:
                self.__print__()

    def play_move(self, move):
        """
        play a move, update state
        """

        r, c = move // 8, move % 8

        assert self.state[r, c] == 0, f"{r}-{c} is already occupied!"

        # get all pieces to be flipped (tbf)
        tbf = self.get_tbf(self.state, self.next_hand_color, move)

        # means current hand is forfeited : we switch
        if len(tbf) == 0:  
            self.next_hand_color *= -1
            tbf = self.get_tbf(self.state, self.next_hand_color, move)

        # either move was illegal, either the game must have ended
        if len(tbf) == 0:
            valids = self.get_valid_moves()
            if len(valids) == 0:
                assert 0, "Both color cannot put piece, game should have ended!"
            else:
                assert 0, "Illegal move!"
        
        # play the move and flip the pieces to be flipped
        self.state[r, c] = self.next_hand_color
        for ff in tbf:
            self.state[ff[0], ff[1]] *= -1

        # hand is switched for next turn
        self.next_hand_color *= -1
        self.history.append(move)
        
    def tentative_move(self, move):
        """
        tentatively put a piece, do nothing to state

        returns 0 if this is not a move at all: occupied or both player have to forfeit (game ended)
        return 1 if regular move
        return 2 if forfeit happens but the opponent can drop piece at this place
        """

        r, c = move // 8, move % 8

        if not self.state[r, c] == 0:
            return 0

        color = self.next_hand_color
        tbf = self.get_tbf(self.state, color, move)

        if len(tbf) != 0:
            return 1
        
        # means current hand is forfeited : we switch
        else:
            color *= -1
            tbf = self.get_tbf(self.state, self.next_hand_color, move)

            if len(tbf) == 0:
                return 0
            else:
                return 2
        
    def get_valid_moves(self):
        """
        Returns all the valid (legal) moves given the current state of the game
        """
        
        regular_moves = []
        forfeit_moves = []

        for move in range(64):
            x = self.tentative_move(move)
            # x = 0 : not a valid move OR both player have to forfeit
            # x = 1 : valid move
            # x = 2 : valid move, but for the other player (ie, the current player forfeits)

            if x == 1:
                regular_moves.append(move)
            elif x == 2:
                forfeit_moves.append(move)

        if len(regular_moves):
            return regular_moves
        elif len(forfeit_moves):
            return forfeit_moves
        else:
            return []
    
    def __print__(self):
        """
        Prints the current state of the game, as well as all the moves.
        """

        print("-" * 20)

        #print([move_to_str(move) for move in self.history])
        print([move for move in self.history])

        a = "abcdefgh"
        for k, row in enumerate(self.state.tolist()):
            row_to_print = []
            for el in row:
                if el == -1:
                    row_to_print.append("O") # white
                elif el == 0:
                    row_to_print.append(" ") # empty
                else:
                    row_to_print.append("X") # black

            print(" ".join([a[k]] + row_to_print))

        row_to_print = [str(k) for k in range(1, 9)]
        print(" ".join([" "] + row_to_print))

        print("-" * 20)

"""
DATASET PART : https://github.com/alxndrTL/othello_mamba/blob/main/data.py
"""


class OthelloDataset(torch.utils.data.IterableDataset):
    def __init__(self, dir: str = "data/train", seed: int = None):
        # dir contains the .bin files created by prepare_data.py
        # each files contains some numbers (around 100K) of tokenized games, each of len 60

        # seed is used by create_data_probing.py to get the same batches when collecting activations and boards
        # setting a seed allows to sample the same batches when collecting games from two different endpoints
        # ie for i, data in enumerate(loader_val) will give the same batches when called multiple times

        super().__init__()

        self.dir = dir
        self.seed = seed

    def __iter__(self):
        # executed by each worker, when data are requested
        # returns one game (ie one training example)

        # every .bin files is a 60*N array, N being the number of games per file (approx. 100K)
        chunks_files = [os.path.join(self.dir, file) for file in os.listdir(self.dir) if file.endswith('.bin')]

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        rng = random.Random(123456 + worker_id)

        if self.seed:
            rng_numpy = np.random.default_rng(self.seed)
        else:
            rng_numpy = np.random.default_rng()

        while True:
            rng.shuffle(chunks_files)
            for chunk_file in chunks_files:
                chunk = np.memmap(chunk_file, dtype=np.int8, mode='r') # read a .bin file
                num_games = chunk.shape[0] // 60

                game_start_indices = 60 * np.arange(num_games) # get all the indices on which games start (all games are padded to a lenght of 60)
                rng_numpy.shuffle(game_start_indices)

                for indice in game_start_indices:
                    start = indice
                    end = start + 60
                    
                    # as the tokenized move are from -1 to 63, we feed to the model 0 to 64 (index -1 should not by used with nn.Embedding)
                    data = torch.from_numpy(chunk[start:end].copy()) + 1
                    x = data[:-1].int() # classic shifting
                    y = data[1:].long() # long() is necessary for the CE loss

                    yield x, y


class ProbingDataset(torch.utils.data.IterableDataset):
    def __init__(self, dir_activations: str, dir_boards: str):
        super().__init__()

        self.dir_activations = dir_activations
        self.dir_boards = dir_boards

    def __iter__(self):

        files_activations = sorted([os.path.join(self.dir_activations, file) for file in os.listdir(self.dir_activations) if file.endswith('.npy')])
        files_boards = sorted([os.path.join(self.dir_boards, file) for file in os.listdir(self.dir_boards) if file.endswith('.npy')])

        files_indices = list(range(len(files_activations)))
        rng = random.Random()
        rng.shuffle(files_indices)

        while True:
            for index in files_indices:
                activations = np.load(files_activations[index]) # (B, 59, d_model) we only get games of len 59 because the model only sees 59 moves as input
                boards = np.load(files_boards[index]) # (B, 59, 8*8)

                activations = activations.reshape(-1, activations.shape[2]) # (B*59, d_model)
                boards = boards.reshape(-1, boards.shape[2]) # (B*59, 8*8)

                sample_indices = list(range(activations.shape[0]))
                rng.shuffle(sample_indices)
                for sample_index in sample_indices:
                    yield activations[sample_index], boards[sample_index]
