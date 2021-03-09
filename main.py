from players import player
from hex_skeleton import HexBoard
from trueskill import Rating, quality_1vs1, rate_1vs1
import numpy as np
import searcher
import matplotlib.pyplot as plt

player1_ratings = Rating()
player2_ratings = Rating()


def play(player1, player2, board,verbose = True):
 
    global player1_ratings
    global player2_ratings

    turn = 0 # turn counter
    
    # while the game is not finished
    while not board.game_over:
        
        turn += 1 # increase turn after a round of the game
        searcher.d3_rt_lst.append(searcher.d3_run_time)
        searcher.d4_rt_lst.append(searcher.d4_run_time)
        searcher.d3_run_time = 0 # and reset run time
        searcher.d4_run_time = 0 
        
        
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

def reset_counter():
    searcher.d3_total_cutoff = 0
    searcher.d4_total_cutoff = 0

    searcher.d3_beta_cutoff = 0
    searcher.d3_alpha_cutoff = 0

    searcher.d4_beta_cutoff = 0
    searcher.d4_alpha_cutoff = 0

#------1. Alpha Beta Experiments----------------------------------------------
# since no pie rule implemented, 1st player has advantage, so order matters
# therefore agents needs to be initalized in reversed play starting order as well

# Agents Matchup 1: [depth = 3, policy = random] vs.  [depth = 3, policy = dijkstra]

# Agents Matchup 2: [depth = 3, policy = random] vs.  [depth = 4, policy = dijkstra]

# Agents Matchup 3: [depth = 3, policy = dijkstra] vs.  [depth = 4, policy = dijkstra]

print("[Initalized Player Matchup: Experiment 1]") 
print("[---------------------------------------------------------]")
res_lst_1 = []
res_elo_1 = []
res_elo_2 = []
res_elo_1.append(player1_ratings.mu)
res_elo_2.append(player2_ratings.mu)  

n_games = 25 # number of games
size = 5 # board size
board = HexBoard(size) # init board 


player_1_lst = [
                player(agent = 'AI', 
                  color = HexBoard.BLUE,
                  policy = "random"),
                
                player(agent = 'AI', 
                  color = HexBoard.BLUE,
                  policy = "random"), 
                
                player(agent = "AI", 
                  color = HexBoard.BLUE,
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
                  depth =  4, switch = True),
                
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "alphabeta",
                  eval = "dijkstra", 
                  depth =  4, switch = True)
                ]

print("[Started - Experiment 1: Model Comparison Alpha Beta . . .]")
print("[---------------------------------------------------------]")

for i in range(len(player_1_lst)):
    player_1 = player_1_lst[i]
    player_2 = player_2_lst[i]
    
    for j in range(n_games):
        board = HexBoard(size)
        play(player_1, player_2, board, verbose = False)
        
        res_elo_1.append(player1_ratings.mu)
        res_elo_2.append(player2_ratings.mu)
    
    print(player1_ratings)
    print(player2_ratings)
    player1_ratings = Rating()
    player2_ratings = Rating()
    res_elo_1.append(player1_ratings.mu)
    res_elo_2.append(player2_ratings.mu)  
    d3_avg_cutoff = searcher.d3_total_cutoff // n_games  
    d3_avg_beta = searcher.d3_beta_cutoff // n_games  
    d3_avg_alpha = searcher.d3_alpha_cutoff // n_games  
    
    d4_avg_cutoff = searcher.d4_total_cutoff // n_games  
    d4_avg_beta = searcher.d4_beta_cutoff // n_games  
    d4_avg_alpha = searcher.d4_alpha_cutoff // n_games 
    
    res_lst_1.extend([d3_avg_alpha, d3_avg_beta,d3_avg_cutoff ,
                      d4_avg_alpha, d4_avg_beta, d4_avg_cutoff ])
    reset_counter()
    d3_avg_cutoff = 0
    d4_avg_cutoff = 0

    d3_avg_beta = 0
    d3_avg_alpha = 0

    d4_avg_beta = 0
    d4_avg_alpha = 0


# plot runtime    
# set n_games = 1 and plot speed per turn and comment out first 2 player
# matchups ins player 1 and player 2 and run for each size in turn 

#searcher.d3_rt_lst = [i for i in searcher.d3_rt_lst if i != 0]
#searcher.d4_rt_lst = [i for i in searcher.d4_rt_lst if i != 0]

# plt.figure(figsize=(8, 6))
# plt.plot(np.array(searcher.d3_rt_lst), linewidth=3)
# plt.plot(np.array(searcher.d4_rt_lst), linewidth=3)
# plt.xticks(np.arange(len(searcher.d3_rt_lst)), np.arange(1, len(searcher.d3_rt_lst)+1))
# plt.title('Runtime per Turn: depth 3 vs. depth 4', fontsize = 20)
# plt.xlabel('Turn', fontsize = 18)
# plt.ylabel('Time in Seconds', fontsize = 18)
# plt.legend(['p1: depth 3 ', 'p2: depth 4'], 
#             prop={'size': 18}, frameon=False)
# plt.show()
# searcher.d3_rt_lst.clear()
# searcher.d4_rt_lst.clear()


# Plot Cutoffs
label = ['M1: alpha beta d=3', 'M2: alpha beta d=4', 'M3:alpha beta d=3', 
          'M3: alpha beta d=4']
data = [ 
        res_lst_1[0:3],
        res_lst_1[9:12],
        res_lst_1[12:15],
        res_lst_1[15:18]
        ]

numpy_array = np.array(data)
transpose = numpy_array.T
transpose_list = transpose.tolist()
X = np.arange(4)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_axes([0,0,1,1])
ax.bar(X + - 0.25, transpose_list[0], color = 'b', width = 0.25)
ax.bar(X + 0.00, transpose_list[1], color = 'g', width = 0.25)
ax.bar(X + 0.25, transpose_list[2], color = 'r', width = 0.25)
plt.xticks(X, label)
ax.set_xticklabels(label, rotation=0, fontsize=12, fontweight='bold')
ax.legend(['Alpha', 'Beta' , 'Total'], prop={'size': 20}, frameon=False)
plt.title('Histogram: Averaged alpha beta cutoffs', fontsize = 22)
plt.xlabel('Type of agent', fontsize = 20)
plt.ylabel('Number of cutoffs', fontsize = 20)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(res_elo_1[0:26], linewidth=3)
plt.plot(res_elo_2[0:26], linewidth=3)
plt.title('Random policy vs. Alpha beta depth 3', fontsize = 20)
plt.xlabel('Number of Games', fontsize = 18)
plt.ylabel('Mean Elo', fontsize = 18)
plt.legend(['p1: random ', 'p2: alpha beta'], 
           prop={'size': 18}, frameon=False)
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(res_elo_1[26:52], linewidth=3)
plt.plot(res_elo_2[26:52], linewidth=3)
plt.title('Random policy vs. Alpha beta depth 4', fontsize = 20)
plt.xlabel('Number of Games', fontsize = 18)
plt.ylabel('Mean Elo', fontsize = 18)
plt.legend(['p1: random ', 'p2: alpha beta'], 
           prop={'size': 18}, frameon=False)
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(res_elo_1[52:78], linewidth=3)
plt.plot(res_elo_2[52:78], linewidth=3)
plt.title('Alpha beta depth 3 vs. Alpha beta depth 4', fontsize = 20)
plt.xlabel('Number of Games', fontsize = 18)
plt.ylabel('Mean Elo', fontsize = 18)
plt.legend(['p1: depth 3 ', 'p2: depth 4'], 
           prop={'size': 18}, frameon=False)
plt.show()


print("[Initalized Reverse-Player Matchup: Experiment 1]")
# reverse player 1 and player 2 since order matters for outcome
player1_ratings = Rating()
player2_ratings = Rating()
res_elo_1.clear()
res_elo_2.clear()
res_elo_1.append(player1_ratings.mu)
res_elo_2.append(player2_ratings.mu)


player_1_lst = [
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "alphabeta",
                  eval = "dijkstra", 
                  depth =  3),
                
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "alphabeta",
                  eval = "dijkstra", 
                  depth =  4, switch = True),
                
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "alphabeta",
                  eval = "dijkstra", 
                  depth =  4, switch = True)
                ]


player_2_lst = [
                player(agent = 'AI', 
                  color = HexBoard.BLUE,
                  policy = "random"),
                
                player(agent = 'AI', 
                  color = HexBoard.BLUE,
                  policy = "random"), 
                
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "alphabeta",
                  eval = "dijkstra", 
                  depth =  3)
                ]
print("[Started - Experiment 1: Reverse Player Order Model Comparison Alpha Beta . . .]")
print("[---------------------------------------------------------]")
for i in range(len(player_1_lst)):
    player_1 = player_1_lst[i]
    player_2 = player_2_lst[i]
    
    for j in range(n_games):
        board = HexBoard(size)
        play(player_1, player_2, board, verbose = False)
        
        res_elo_1.append(player1_ratings.mu)
        res_elo_2.append(player2_ratings.mu)
    
    d3_avg_cutoff = searcher.d3_total_cutoff // n_games  
    d3_avg_beta = searcher.d3_beta_cutoff // n_games  
    d3_avg_alpha = searcher.d3_alpha_cutoff // n_games  
    
    d4_avg_cutoff = searcher.d4_total_cutoff // n_games  
    d4_avg_beta = searcher.d4_beta_cutoff // n_games  
    d4_avg_alpha = searcher.d4_alpha_cutoff // n_games 
    
    res_lst_1.extend([d3_avg_alpha, d3_avg_beta,d3_avg_cutoff ,
                      d4_avg_alpha, d4_avg_beta, d4_avg_cutoff ])
    reset_counter()
    d3_avg_cutoff = 0
    d4_avg_cutoff = 0

    d3_avg_beta = 0
    d3_avg_alpha = 0

    d4_avg_beta = 0
    d4_avg_alpha = 0

    print(player1_ratings)
    print(player2_ratings)
    player1_ratings = Rating()
    player2_ratings = Rating()
    res_elo_1.append(player1_ratings.mu)
    res_elo_2.append(player2_ratings.mu) 


plt.figure(figsize=(8, 6))
plt.plot(res_elo_1[0:26], linewidth=3)
plt.plot(res_elo_2[0:26], linewidth=3)
plt.title('Alpha beta depth 3 vs. Random  policy', fontsize = 20)
plt.xlabel('Number of Games', fontsize = 18)
plt.ylabel('Mean Elo', fontsize = 18)
plt.legend(['p1: alpha beta ', 'p2: random'], 
           prop={'size': 18}, frameon=False)
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(res_elo_1[26:52], linewidth=3)
plt.plot(res_elo_2[26:52], linewidth=3)
plt.title('Alpha beta depth 4 vs. Random policy ', fontsize = 20)
plt.xlabel('Number of Games', fontsize = 15)
plt.ylabel('Mean Elo', fontsize = 18)
plt.legend(['p1: alpha beta ', 'p2: random'], 
           prop={'size': 18}, frameon=False)
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(res_elo_1[52:78], linewidth=3)
plt.plot(res_elo_2[52:78], linewidth=3)
plt.title(' Alpha beta depth 4 vs. Alpha beta depth 3', fontsize = 20)
plt.xlabel('Number of Games', fontsize = 18)
plt.ylabel('Mean Elo', fontsize = 18)
plt.legend(['p1: depth 4 ', 'p2: depth 3'], 
           prop={'size': 18}, frameon=False)
plt.show()    

print("[Completed - Experiment 1: Model Comparison Alpha Beta.]")
print("[---------------------------------------------------------]")
#------2. Iterative Deepening and Transposition Table Experiments-------------

# Same Matchup as before but with ITDD should perform similar 


print("[Initalized Player Matchup: Experiment 2]")
print("[---------------------------------------------------------]")
n_games = 25 # number of games
size = 5 # board size
board = HexBoard(size) # init board 
player1_ratings = Rating()
player2_ratings = Rating()
res_elo_1 = []
res_elo_2 = []
player1_ratings = Rating()
player2_ratings = Rating()
res_elo_1.append(player1_ratings.mu)
res_elo_2.append(player2_ratings.mu)  

player_1_lst = [
                player(agent = 'AI', 
                  color = HexBoard.BLUE,
                  policy = "random"),
                
                player(agent = 'AI', 
                  color = HexBoard.BLUE,
                  policy = "random"), 
                
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "iterdeep",
                  eval = "dijkstra", 
                  depth =  3, use_tt=True)
                ]

player_2_lst = [
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "iterdeep",
                  eval = "dijkstra", 
                  depth =  3, use_tt = True),
                
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "iterdeep",
                  eval = "dijkstra", 
                  depth =  4, use_tt = True,
                  switch = True),
                
                player(agent = "AI", 
                  color = HexBoard.RED,
                  policy = "iterdeep",
                  eval = "dijkstra", 
                  depth =  4, use_tt = True, 
                  switch = True)
                ]



print("[Started - Experiment 2: Model Comparison Table Enhancements . . .]")
print("[---------------------------------------------------------]")
for i in range(len(player_1_lst)):
    player_1 = player_1_lst[i]
    player_2 = player_2_lst[i]
    
    for j in range(n_games):
        board = HexBoard(size)
        play(player_1, player_2, board, verbose = False)
        
        res_elo_1.append(player1_ratings.mu)
        res_elo_2.append(player2_ratings.mu)
    
    print(player1_ratings)
    print(player2_ratings)
    player1_ratings = Rating()
    player2_ratings = Rating()
    res_elo_1.append(player1_ratings.mu)
    res_elo_2.append(player2_ratings.mu) 



plt.figure(figsize=(8, 6))
plt.plot(res_elo_1[0:26], linewidth=3)
plt.plot(res_elo_2[0:26], linewidth=3)
plt.title('Random policy vs. IDTT depth 3', fontsize = 20)
plt.xlabel('Number of Games', fontsize = 18)
plt.ylabel('Mean Elo', fontsize = 18)
plt.legend(['p1: random ', 'p2: alpha beta'], 
           prop={'size': 18}, frameon=False)
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(res_elo_1[26:52], linewidth=3)
plt.plot(res_elo_2[26:52], linewidth=3)
plt.title('Random policy vs. IDDT depth 4', fontsize = 20)
plt.xlabel('Number of Games', fontsize = 18)
plt.ylabel('Mean Elo', fontsize = 18)
plt.legend(['p1: random ', 'p2: alpha beta'], 
           prop={'size': 18}, frameon=False)
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(res_elo_1[52:78], linewidth=3)
plt.plot(res_elo_2[52:78], linewidth=3)
plt.title(' IDTT depth 3 vs. IDTT depth 4', fontsize = 20)
plt.xlabel('Number of Games', fontsize = 18)
plt.ylabel('Mean Elo', fontsize = 18)
plt.legend(['p1: depth 3 ', 'p2: depth 4'], 
           prop={'size': 18}, frameon=False)
plt.show()

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

# As such the  match up list is:

# Agents Matchup 1: [depth = 4, policy = idtt] vs. [policy = mcts, N=iter_max=100, C= sqrt(2)]

# Agents Matchup 2: [depth = 4, policy = idtt] vs. [policy = mcts, N=iter_max=1, C= sqrt(2)]

# Agents Matchup 3: [depth = 4, policy = idtt] vs. [policy = mcts, N=iter_max=1000, C = 1000]
print("[Started - Experiment 3: Model Comparison Monte Carlo Tree Search . . .]")
print("[---------------------------------------------------------]")

n_games = 25 # number of games
size = 5 # board size
board = HexBoard(size) # init board 

res_elo_1 = []
res_elo_2 = []
player1_ratings = Rating()
player2_ratings = Rating()
res_elo_1.append(player1_ratings.mu)
res_elo_2.append(player2_ratings.mu)  


player_1_lst = [
                player(agent = 'AI', 
                  color = HexBoard.BLUE,
                  policy = "iterdeep",
                  depth = 4, use_tt = True), 
                
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "iterdeep",
                  eval = "dijkstra", 
                  depth =  4, use_tt = True),
                
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "iterdeep",
                  eval = "dijkstra", 
                  depth =  4, use_tt = True)
                ]

player_2_lst = [

                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "mcts", 
                  max_iter = 2500, 
                  C = np.sqrt(2)),
                
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "mcts", 
                  max_iter = 1,
                  C = np.sqrt(2)),
                
                player(agent = "AI", 
                  color = HexBoard.BLUE,
                  policy = "mcts", 
                  max_iter = 2500, 
                  C = 1000),
                ]

for i in range(len(player_1_lst)):
    player_1 = player_1_lst[i]
    player_2 = player_2_lst[i]
    
    for j in range(n_games):
        board = HexBoard(size)
        play(player_1, player_2, board, verbose = False)
        
        res_elo_1.append(player1_ratings.mu)
        res_elo_2.append(player2_ratings.mu)
    
    print(player1_ratings)
    print(player2_ratings)
    player1_ratings = Rating()
    player2_ratings = Rating()
    res_elo_1.append(player1_ratings.mu)
    res_elo_2.append(player2_ratings.mu) 
    
plt.figure(figsize=(8, 6))
plt.plot(res_elo_1[0:26], linewidth=3)
plt.plot(res_elo_2[0:26], linewidth=3)
plt.title('IDTT depth 4 vs. MCTS N = 2500, C=' r'$\sqrt{2}$', fontsize = 20)
plt.xlabel('Number of Games', fontsize = 18)
plt.ylabel('Mean Elo', fontsize = 18)
plt.legend(['p1: IDTT', 'p2: MCTS'], 
           prop={'size': 18}, frameon=False)
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(res_elo_1[26:52], linewidth=3)
plt.plot(res_elo_2[26:52], linewidth=3)
plt.title('IDTT depth 4 vs. MCTS N = 1, C=' r'$\sqrt{2}$', fontsize = 20)
plt.xlabel('Number of Games', fontsize = 18)
plt.ylabel('Mean Elo', fontsize = 18)
plt.legend(['p1: IDTT ', 'p2: MCTS'], 
           prop={'size': 18}, frameon=False)
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(res_elo_1[52:78], linewidth=3)
plt.plot(res_elo_2[52:78], linewidth=3)
plt.title('IDTT depth 4 vs. MCTS N = 2500, C= 1000', fontsize = 20)
plt.xlabel('Number of Games', fontsize = 18)
plt.ylabel('Mean Elo', fontsize = 18)
plt.legend(['p1: IDTT ', 'p2: MCTS'], 
           prop={'size': 18}, frameon=False)
plt.show()
