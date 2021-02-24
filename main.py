from players import player
from hex_skeleton import HexBoard
from trueskill import Rating, quality_1vs1, rate_1vs1

player1_ratings = Rating()
player2_ratings = Rating()

# initalizing the board
size = 4
board = HexBoard(size)


# initialize the player type
player_1 = player(playerType = 'A.I', color = HexBoard.RED, policy = 'random',
                  eval = 'dijkstra', depth = 2, time_limit = None)
# player_1 = player(playerType = 'Human', color = HexBoard.RED) 
player_2 = player(playerType = 'A.I', color = HexBoard.BLUE, policy = 'alphabeta', 
                  eval = 'dijkstra', depth = 2, time_limit = 0.5)


def play(player1,player2,board,verbose = True):
 
    global player1_ratings
    global player2_ratings

    # initalize turn counter
    turn = 0
    
    # While the game is not finished
    while not board.game_over:
        
        #increment turn after every round of the game
        turn += 1
        
        if verbose == True:
            print("--------turn:{}".format(turn))
            print("------------Player 1:{}[{}] turn-------------".format(player1.playerType,player1.policy))
            board.print()
        # calling player-1 move    
        player1.move(board,verbose=verbose)
        
        # Check if player-1 won the game
        if board.check_win(player1.color):
            if verbose == True:
                board.print()
            print("The Game is Over. Player 1: {}[{}] won the game".format(player1.playerType,player1.policy))
            player1_ratings,player2_ratings = rate_1vs1(player1_ratings,player2_ratings)
            break
        
        if verbose == True:
            print("------------Player 2:{}[{}] turn-------------".format(player2.playerType, player2.policy))
            board.print()
        # calling player-2 move
        player2.move(board,verbose=verbose)

        #check if player-2 won the game
        if board.check_win(player2.color):
            if verbose == True:
                
                board.print()
            print("The Game is Over. Player 2[{}] won the game".format(player2.playerType))
            player2_ratings,player1_ratings = rate_1vs1(player2_ratings,player1_ratings)
            break

games = 25
for i in range(games):
    size = 4
    board = HexBoard(size)
    play(player_1,player_2,board,verbose=False)

print(player1_ratings)
print(player2_ratings)