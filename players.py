from searcher import Searcher
import numpy as np 

class player:

    def __init__(self, agent, color, policy = None, 
                 depth = None, eval = None, time_limit = 1, use_tt = False, max_iter = 100, C = np.sqrt(2),
                 switch = False):

        self.agent = agent
        self.color = color
        self.policy = policy
        self.eval = eval
        self.depth = depth
        self.time_limit = time_limit
        self.use_tt = use_tt
        self.max_iter = max_iter
        self.C = C
        self.switch = switch 

    def move(self, board,verbose):

        # condition if player is human
        if self.agent == "Human":
            board.print()
            # while true human player has to input a valid response
            while True:
                # response format is a tuple 
                try:
                    move = tuple(map(int,(input('Please indicate your move[response-format:("x-axis value, y-axis value")]: ')).split(',')))
                
                # if the move is invalid i.e. not possible then prompt
                except ValueError:
                    print("This move is not possible. Please try another one!")
                
                # check if given input is available
                else:
                    try:
                        if board.is_empty(move):
                            # place hexagon on board
                            board.place(move, self.color)
                            break
                        else:
                            print("This tile has already been taken by the enemy player. The move is not possible")
                    
                    except KeyError:

                        print("This tile has already been taken by the enemy player. The move is not possible")


        if self.agent == 'AI':
            # Use the searcher to find the best move and place it on the board

            search = Searcher(depth = self.depth, policy = self.policy, board = board, 
                              color = self.color, eval = self.eval,
                              time_limit = self.time_limit, use_tt = self.use_tt,
                              max_iter = self.max_iter, C = self.C, switch = self.switch)

            best_move = search.find_bm()
            if verbose:
                print("AI[{}] make a place at: {}".format(self.policy,best_move))
            board.place(best_move,self.color)