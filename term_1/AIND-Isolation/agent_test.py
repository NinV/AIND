"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

import isolation
import game_agent
from sample_players import *



class alpha_beta_test(unittest.TestCase):
    """Unit tests for Minimax agents"""

    def test_optimalMove_1(self):

        board_width = 9
        board_height = 9
        search_depth = 1

        self.player1 = game_agent.AlphaBetaPlayer(search_depth=search_depth, score_fn=improved_score)
        self.player2 = game_agent.AlphaBetaPlayer(search_depth=search_depth, score_fn=improved_score)
        self.game = isolation.Board(self.player1, self.player2, board_width, board_height)

        test_game = self.game.copy()
        test_game._board_state = [0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 1, 1, 1, 0, 0, 0,
                                  0, 0, 1, 1, 0, 1, 1, 0, 0,
                                  0, 0, 1, 1, 0, 0, 1, 0, 0,
                                  0, 0, 1, 1, 0, 1, 1, 0, 0,
                                  0, 1, 0, 1, 1, 1, 0, 0, 0,
                                  0, 0, 0, 1, 0, 0, 0, 0, 0,
                                  0, 0, 1, 0, 0, 0, 0, 0, 0,
                                  0, 74, 22]
        optimal_move_list = [(2, 1), ]
        print(test_game.to_string())
        # dummy function which always return positive value
        time_left = lambda: 150
        selection_move = test_game._active_player.get_move(test_game, time_left=time_left)
        print("Selection move:\n", selection_move)
        self.assertTrue(selection_move in optimal_move_list)


if __name__ == '__main__':
    unittest.main()


