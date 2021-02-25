from players import player
from hex_skeleton import HexBoard
from trueskill import Rating, quality_1vs1, rate_1vs1

player1_ratings = Rating()
player2_ratings = Rating()


size = 4 # board size
board = HexBoard(size) # init board 


# initialize the player type
player_1 = player(playerType = 'A.I', color = HexBoard.RED, policy = 'random',
                  eval = 'dijkstra', depth = 2, time_limit = None)
# player_1 = player(playerType = 'Human', color = HexBoard.RED) 
# player_2 = player(playerType = 'A.I', color = HexBoard.BLUE, policy = 'alphabeta', 
#                  eval = 'dijkstra', depth = 2, time_limit = None)
player_2 = player(playerType = 'A.I', color = HexBoard.BLUE, policy = 'i_deep', 
                  eval = 'dijkstra', depth = 2, time_limit = 0.5)

def play(player1,player2,board,verbose = True):
 
    global player1_ratings
    global player2_ratings

    # initalize turn counter
    turn = 0
    
    # While the game is not finished
    while not board.game_over:
        
        turn += 1 # increase turn after a round of the game
        
        if verbose == True:
            print("turn:{}".format(turn))
            print("Player 1:{}[{}]".format(player1.playerType,player1.policy))
            board.print()
        
        player1.move(board,verbose=verbose) # player_1 moves
        
        # if player _1 won game then :
        if board.check_win(player1.color):
            if verbose == True:
                board.print()
            print("The Game is Over. Player 1: {}[{}] won the game".format(player1.playerType,player1.policy))
            player1_ratings,player2_ratings = rate_1vs1(player1_ratings,player2_ratings)
            break
        
        if verbose == True:
            print("Player 2:{}[{}]".format(player2.playerType, player2.policy))
            board.print()
        
        player2.move(board,verbose=verbose) # player_2 moves

        # if player_2 won game then :
        if board.check_win(player2.color):
            if verbose == True:
                
                board.print()
            print("The Game is Over. Player 2[{}] won the game".format(player2.playerType))
            player2_ratings,player1_ratings = rate_1vs1(player2_ratings,player1_ratings)
            break

n_games = 4
for i in range(n_games):
    size = 4
    board = HexBoard(size)
    play(player_1,player_2,board,verbose=False)

print(player1_ratings)
print(player2_ratings)