from searcher import searcher


class player:

    def __init__(self, playerType, color, policy , eval,
                 depth , time_limit):

        self.playerType = playerType
        self.color = color
        self.policy = policy
        self.eval = eval
        self.depth = depth
        self.time_limit = time_limit


    def move(self, board,verbose):

        # condition if player is human
        if self.playerType == "Human":
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
                            board.place(move,self.color)
                            break
                        else:
                            print("This tile has already been taken by the enemy player. The move is not possible")
                    
                    except KeyError:

                        print("This tile has already been taken by the enemy player. The move is not possible")


        if self.playerType == 'A.I':
            # Use the searcher to find the best move and place it on the board

            search = searcher(depth = self.depth, policy = self.policy, board = board, 
                              color = self.color, eval = self.eval, 
                              #iter_deep = self.iter_deep, 
                              time_limit = self.time_limit)

            best_move = search.findBestMove()
            if verbose:
                print("A.I[{}] make a place at: {}".format(self.policy,best_move))
            board.place(best_move,self.color)