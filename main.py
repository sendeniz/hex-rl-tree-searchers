from players import player
from hex_skeleton import HexBoard
from trueskill import Rating, quality_1vs1, rate_1vs1
import numpy as np

player1_ratings = Rating()
player2_ratings = Rating()


n_games = 25 # number of games
size = 5 # board size
board = HexBoard(size) # init board 

def play(player1, player2, board,verbose = True):
 
    global player1_ratings
    global player2_ratings

    turn = 0 # turn counter
    
    # while the game is not finished
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
            print("The Game is Over. Player 1: {}[{}] won the game".format(player1.agent,
                                                                            player1.policy))
            
            player1_ratings,player2_ratings = rate_1vs1(player1_ratings, player2_ratings)
            break
        
        if verbose == True:
            print("Player 2:{}[{}]".format(player2.agent, player2.policy))
            board.print()
        
        player2.move(board,verbose=verbose) # player_2 moves

        # if player_2 won game then :
        if board.check_win(player2.color):
            if verbose == True:
                board.print()
            print("The Game is Over. Player 2: {}[{}] won the game".format(player2.agent, 
                                                                           player2.policy))
            
            player2_ratings, player1_ratings = rate_1vs1(player2_ratings,player1_ratings)
            break

#------1. Alpha Beta Experiments----------------------------------------------
# since no pie rule implemented, 1st player has advantage, so order matters
# therefore agents needs to be initalized in reversed play starting order as well

# Agents Matchup 1: [depth = 3, policy = random] vs.  [depth = 3, policy = dijkstra]

# Agents Matchup 2: [depth = 3, policy = random] vs.  [depth = 4, policy = dijkstra]

# Agents Matchup 3: [depth = 3, policy = dijkstra] vs.  [depth = 4, policy = dijkstra]


player_1_lst = [
                player(agent = 'AI', 
                  color = HexBoard.RED,
                  policy = "random",
                  depth = 3),
                
                player(agent = 'AI', 
                  color = HexBoard.RED,
                  policy = "random",
                  depth = 3), 
                
                player(agent = "AI", 
                  color = HexBoard.RED,
                  policy = "alphabeta",
                  eval = "dijkstra", 
                  depth =  3)
                ]

player_2_lst = [
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "alphabeta",
                  eval = "dijkstra", 
                  depth =  3),
                
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "alphabeta",
                  eval = "dijkstra", 
                  depth =  4),
                
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "alphabeta",
                  eval = "dijkstra", 
                  depth =  4)
                ]

print("[Started - Experiment 1: Model Comparison Alpha Beta . . .]")
print("[---------------------------------------------------------]")
for i in range(len(player_1_lst)):
    player_1 = player_1_lst[i]
    player_2 = player_2_lst[i]
    
    for j in range(n_games):
        board = HexBoard(size)
        play(player_1, player_2, board, verbose = False)
    print(player1_ratings)
    print(player2_ratings)

# reverse player 1 and player 2 since order matters for outcome
for i in range(len(player_1_lst)):
    player_1 = player_2_lst[i]
    player_2 = player_1_lst[i]
    
    for k in range(n_games):
        board = HexBoard(size)
        play(player_1, player_2, board, verbose = False)
    print(player1_ratings)
    print(player2_ratings)
    

print("[Completed - Experiment 1: Model Comparison Alpha Beta.]")
print("[---------------------------------------------------------]")
#------2. Iterative Deepening and Transposition Table Experiments-------------

# Reasonable comparision is, whether iterative deepening with and without 
# transpositon tables outperforms random policy or regular alpha beta
# As such the  match up list is:

# Agents Matchup 1: [depth = 3, policy = random] vs. [depth = 3, policy = alpha beta + iterdeep]

# Agents Matchup 2: [depth = 3, policy = random] vs. [depth = 3, policy = alpha beta + iterdeep + tt]

# Agents Matchup 3: [depth = 3, policy = alphabeta] vs. [depth = 4, policy = alpha beta + iterdeep]

# Agents Matchup 4: [depth = 4, policy = alphabeta] vs. [depth = 4, policy = alpha beta + iterdeep + tt]

# Optional: Agents Matchup 5: [depth = 3, policy = alpha beta +iterdeep] vs. [depth = 4, policy = alpha beta + iterdeep + tt]

print("[Started - Experiment 2: Model Comparison Table Enhancements . . .]")
print("[---------------------------------------------------------]")
player_1_lst = [
                player(agent = 'AI', 
                  color = HexBoard.RED,
                  policy = "random",
                  depth = 3),
                
                player(agent = 'AI', 
                  color = HexBoard.RED,
                  policy = "random",
                  depth = 3), 
                
                player(agent = "AI", 
                  color = HexBoard.RED,
                  policy = "alphabeta",
                  eval = "dijkstra", 
                  depth =  3),
                
                player(agent = "AI", 
                  color = HexBoard.RED,
                  policy = "alphabeta",
                  eval = "dijkstra", 
                  depth =  3)]

player_2_lst = [
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "iterdeep", 
                  time_limit = 1,
                  eval = "dijkstra", 
                  depth =  3),
                
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "iterdeep", 
                  time_limit = 1,
                  eval = "dijkstra", 
                  depth =  3,
                  use_tt = True),
                
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "iterdeep",
                  eval = "dijkstra", 
                  depth =  3),

                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "iterdeep",
                  eval = "dijkstra", 
                  depth =  3, use_tt = True)]


for i in range(len(player_1_lst)):
    player_1 = player_1_lst[i]
    player_2 = player_2_lst[i]
    
    for j in range(n_games):
        board = HexBoard(size)
        play(player_1, player_2, board, verbose = False)
    print(player1_ratings)
    print(player2_ratings)

print("[Started - Experiment 2: Reverse Player Order Model Comparison Table Enhancements . . .]")
print("[---------------------------------------------------------]")
# reverse player 1 and player 2 since order matters for outcome
for i in range(len(player_1_lst)):
    player_1 = player_2_lst[i]
    player_2 = player_1_lst[i]
    
    for k in range(n_games):
        board = HexBoard(size)
        play(player_1, player_2, board, verbose = False)
    print(player1_ratings)
    print(player2_ratings)
    
print("[Completed - Experiment 2: Model Comparison Table Enhancements.]")
print("[---------------------------------------------------------]")

#------3. Monte Carlo Tree Search (MCTS)--------------------------------------

# Reasonable comparision is, whether mcts outperforms, random policy, 
# regular alpha beta and alpha beta iterative deepening with  transpositon tables
 
# As such the  match up list is:

# Agents Matchup 1: [depth = 3, policy = random] vs. [policy = mcts, N=iter_max=100, C= sqrt(2)]

# Agents Matchup 2: [depth = 3, policy = alphabeta] vs. [policy = mcts, N=iter_max=100, C= sqrt(2)]

# Agents Matchup 3: [depth = 3, policy = alpha beta + iterdeep + tt] vs. [policy = mcts, N=iter_max=100, C= sqrt(2)]]

player_1_lst = [
                player(agent = 'AI', 
                  color = HexBoard.RED,
                  policy = "random",
                  depth = 3),
                
                player(agent = 'AI', 
                  color = HexBoard.RED,
                  policy = "alphabeta",
                  depth = 3), 
                
                player(agent = "AI", 
                  color = HexBoard.RED,
                  policy = "iterdeep",
                  eval = "dijkstra", 
                  depth =  3, use_tt = True)
                ]

player_2_lst = [
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "mcts", 
                  max_iter = 100,
                  C = np.sqrt(2)),
                
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "mcts", 
                  max_iter = 100, 
                  C = np.sqrt(2)),
                
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "mcts", 
                  max_iter = 100,
                  C = np.sqrt(2))
                ]

print("[Started - Experiment 3: Model Comparison Monte Carlo Tree Search . . .]")
print("[---------------------------------------------------------]")
for i in range(len(player_1_lst)):
    player_1 = player_1_lst[i]
    player_2 = player_2_lst[i]
    
    for j in range(n_games):
        board = HexBoard(size)
        play(player_1, player_2, board, verbose = False)
    print(player1_ratings)
    print(player2_ratings)

print("[Started - Experiment 3: Reverse Player Order Model Comparison Monte Carlo Tree Search . . .]")
print("[---------------------------------------------------------]")
# reverse player 1 and player 2 since order matters for outcome
for i in range(len(player_1_lst)):
    player_1 = player_2_lst[i]
    player_2 = player_1_lst[i]
    
    for k in range(n_games):
        board = HexBoard(size)
        play(player_1, player_2, board, verbose = False)
    print(player1_ratings)
    print(player2_ratings)

print("[Completed - Experiment 3: Model Comparison Model Comparison Monte Carlo Tree Search.]")
print("[---------------------------------------------------------]")

#---------------Hyperparemter C and N=max_iter Experiments--------------------
# C ontrols explotations/exploration trade off
# N are the number of simulations to be down since MCTS is a sampling method

# Since two hyperparemters grid search is a feasable method to examine effect 
# of two paremters with small, medium and large values while holding the other
# fixed

#------------with C fixed and N=max_iter increasing 
player_1_lst = [
                player(agent = 'AI', 
                  color = HexBoard.RED,
                  policy = "random",
                  depth = 3),
                
                player(agent = 'AI', 
                  color = HexBoard.RED,
                  policy = "alphabeta",
                  depth = 3), 
                
                player(agent = "AI", 
                  color = HexBoard.RED,
                  policy = "iterdeep",
                  eval = "dijkstra", 
                  depth =  3, use_tt = True)
                ]

player_2_lst = [
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "mcts", 
                  max_iter = 1,
                  C = np.sqrt(2)),
                
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "mcts", 
                  max_iter = 10, 
                  C = np.sqrt(2)),
                
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "mcts", 
                  max_iter = 100,
                  C = np.sqrt(2))
                ]

print("[Started - Experiment 4: Hyperparameter N Monte Carlo Tree Search . . .]")
print("[---------------------------------------------------------]")
for i in range(len(player_1_lst)):
    player_1 = player_1_lst[i]
    player_2 = player_2_lst[i]
    
    for j in range(n_games):
        board = HexBoard(size)
        play(player_1, player_2, board, verbose = False)
    print(player1_ratings)
    print(player2_ratings)


#------------with N=max_iter fixed and C increasing
print("[Started - Experiment 4: Hyperparameter C Monte Carlo Tree Search . . .]")
print("[---------------------------------------------------------]")

player_1_lst = [
                player(agent = 'AI', 
                  color = HexBoard.RED,
                  policy = "random",
                  depth = 3),
                
                player(agent = 'AI', 
                  color = HexBoard.RED,
                  policy = "alphabeta",
                  depth = 3), 
                
                player(agent = "AI", 
                  color = HexBoard.RED,
                  policy = "iterdeep",
                  eval = "dijkstra", 
                  depth =  3, use_tt = True)
                ]

player_2_lst = [
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "mcts", 
                  max_iter = 100,
                  C = 0.01),
                
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "mcts", 
                  max_iter = 100, 
                  C = 10),
                
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "mcts", 
                  max_iter = 100,
                  C = 100)
                ]

for i in range(len(player_1_lst)):
    player_1 = player_1_lst[i]
    player_2 = player_2_lst[i]
    
    for j in range(n_games):
        board = HexBoard(size)
        play(player_1, player_2, board, verbose = False)
    print(player1_ratings)
    print(player2_ratings)

print("[Completed - Experiment 4: Hyperparameter N Monte Carlo Tree Search.]")
print("[---------------------------------------------------------]")

print("[Human Player against MCTS AI]")
print("[---------------------------------------------------------]")
player_1_lst = [
                player(agent = 'Human', 
                  color = HexBoard.RED)
                ]

player_2_lst = [
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "mcts", 
                  max_iter = 100,
                  C = 0.01)
                ]

for i in range(len(player_1_lst)):
    player_1 = player_1_lst[i]
    player_2 = player_2_lst[i]
    
    for j in range(n_games):
        board = HexBoard(size)
        play(player_1, player_2, board, verbose = False)
    print(player1_ratings)
    print(player2_ratings)


