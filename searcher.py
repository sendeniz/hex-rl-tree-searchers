import random
import numpy as np
import time
from hex_skeleton import HexBoard

#MAX = 999
#MIN = -999

run_time = 0 
elapsed_time = 0
beta_cutoff = 0
alpha_cutoff = 0
maxDepth = 1

class searcher:

    def __init__(self, depth, policy, board, color, eval, time_limit):
        
        self.depth = depth
        self.policy = policy
        self.board = board
        self.color = color
        self.eval = eval
        #self.iter_deep = iter_deep
        self.time_limit = time_limit
    
    def findBestMove(self):
        
        # dummy evaluation as random move
        if self.policy == "random":
            moveList = getMoveList(self.board)
            return random.choice(moveList)
        
        # alpha beta decision policy for AI
        elif self.policy == "alphabeta":
            __, bm = self.alpha_beta(board = self.board, depth = self.depth, 
                                     alpha = np.NINF, beta = np.Inf, max_player = True, 
                                     color = self.color, 
                                     time_limit = self.time_limit)
            return bm
        
        # alpha beta + transposition tables decision poliy for AI
        elif self.policy == "ttalphabeta":
            __, bm = self.tt_alpha_beta(board = self.board, depth = self.depth, 
                                        alpha = np.NINF, beta = np.Inf, max_player = True, 
                                        color = self.color,
                                        time_limit = self.time_limit)
            return bm
        
        elif self.policy == "i_deep":
            bm = self.iter_deep(board = self.board,
                                        alpha = np.NINF, beta = np.Inf, max_player = True, 
                                        color = self.color,
                                        time_limit = self.time_limit)
            return bm
        
    def heuristicEval(self,board):
        AI_color = self.color
        #board.print()
        graph1 = createGraph(board,AI_color)
        val1 = dijkstra(graph1,'start','end')
        graph2 = createGraph(board,board.get_opposite_color(AI_color))
        val2 = dijkstra(graph2,'start','end')
        #print(val2-val1)
        # instead of simple dijstra, compute dijkstra from both player perspective
        # and gives difference between both 
        return val2-val1, (10,10)
    
    def alpha_beta(self, board, depth, alpha, beta, max_player, color, 
                   time_limit):   
        global alpha_cutoff
        global beta_cutoff
        global run_time
        global elapsed_time 
        
        start_time = time.time()
        bm = None # initalize best moves to store
        
        moveList = getMoveList(board)
        if (depth <= 0) or (board.game_over):
            #board.print()
            return self.heuristicEval(board)
        
        elif max_player: # Max Player
            
            g = np.NINF # best value g
            
            # iterate over c moves 
            #for move in getMoveList(board): # search best move
            for move in moveList:
                board = makeMove(board, move, color)
                
                gc, __ = self.alpha_beta(board, depth - 1, alpha, beta, 
                                         False, time_limit, 
                                         board.get_opposite_color(color))
                board = unMakeMove(board, move)
                #print("gc:",gc)
                #print("g:",g)
                if gc >= g : # if value gc  >= best value g
                    bm = move # save best move information
                g = max(gc,g) # max (value, best value)
                alpha = max(alpha, g) # update alpha

                if alpha >= beta:
                    # global beta_cutoff
                    beta_cutoff +=1
                    #print("beta cutoff:",beta_cutoff)
                    break

        elif not max_player: # Min player
            
            g = np.Inf # best value g
            
            # iterative over c moves
            # for move in getMoveList(board):
            for move in moveList:
                board = makeMove(board,move,color)
                gc, __ = self.alpha_beta(board, depth - 1, alpha, beta, 
                                         True, time_limit, 
                                         board.get_opposite_color(color))
                board = unMakeMove(board,move)
                #print("gc:",gc)
                #print("g:",g)
                if gc <= g: # if value gc <= best value g
                    bm = move # store best move info
                g = min(gc,g) # min(value, best value)
                beta = min(beta, g) # update beta

                if alpha >= beta:
                    #global alpha_cutoff
                    alpha_cutoff +=1
                    #print("alpha cutoff:", alpha_cutoff)
                    break
                
            run_time += time.time() - start_time
            #print("run_time:",run_time)
        return g, bm # return best value and best move
        
    def tt_alpha_beta(self, board, depth, alpha, beta, max_player, color,
                      time_limit):   
        # print("TT Alpha Beta Active")
        global run_time
        start_time = time.time()
        
        d_cal = maxDepth - depth
        (hit, g, ttbm) = lookup(board, d_cal)
        #print((hit, g, ttbm))
        # If hit
        if hit and d_cal >= 2:
            return g, ttbm
            
        bm = None # initalize best moves to store
        
        # If not hit and TT table is empty
        if ttbm is None:
            moveList = getMoveList(board) # , ordering=True)
        # If not hit and TT table has some move associated with it.
        else:
            moveList = [ttbm] + getMoveList(board) #, ordering=True)
        if (depth <= 0) or (board.game_over):
            #board.print()
            return self.heuristicEval(board), bm
        
        elif max_player: # Max Player
            
            g = np.NINF # best value g
            
            # iterate over c moves 
            for move in moveList: # search best move
                board = makeMove(board, move, color)
                
                gc, __ = self.tt_alpha_beta(board, depth-1, alpha, beta, 
                                            False, board.get_opposite_color(color),
                                            time_limit)
                
                if isinstance (gc, tuple) == True:
                    gc_temp = gc[0]
                else :
                    gc_temp = gc
                #gc_temp = gc[0]
                board = unMakeMove(board,move)
               
                # if gc >= g : # if value gc  >= best value g
                if gc_temp > g : # if value gc  >= best value g
                    bm = move # save best move information
                    #g = gc
                    g = gc_temp

                alpha = max(alpha,g) # update alpha

                if alpha >= beta:
                    global beta_cutoff
                    beta_cutoff +=1
                    print("beta cutoff:", beta_cutoff)
                    break

        elif not max_player: # Min player
            
            g = np.Inf # best value g
            
            # iterative over c moves
            for move in moveList:
                board = makeMove(board,move,color)
                board, depth-1, alpha, beta, 

                gc, __ = self.tt_alpha_beta(board, depth-1, alpha, beta, 
                                            True, board.get_opposite_color(color),
                                            time_limit)
                
                if isinstance (gc, tuple) == True:
                    gc_temp = gc[0]
                else :
                    gc_temp = gc
                
                board = unMakeMove(board,move)
                
                # if gc <= g: # if value gc <= best value g
                if gc_temp > g : # if value gc  >= best value g
                    bm = move # store best move info
                    # g = gc
                    g = gc_temp
                
                beta = min(beta, g) # update beta
                if alpha >= beta:
                    global alpha_cutoff
                    alpha_cutoff +=1
                    print("alpha cutoff:",alpha_cutoff)
                    break
            
            run_time += time.time() - start_time
            #print("run_time:",run_time)
        store(board, g, d_cal, bm)
        return g, bm # return best value and best move
    
    def iter_deep(self, board, max_player, alpha, beta, color, time_limit=2, TT=False):
        global maxDepth
    
        # Start timer
        start = time.time()
        # initialize depth
        depth = 1
        # Initialize prevBestmove
        prev_bm = tuple()
    
        while True:
            if TT:
                # find the bestMove iteratively for differnt depths but with use of transposition tables
                __, bm = self.tt_alpha_beta(board, depth, alpha, beta, max_player, color,
                          time_limit)
    
            else:
                # find the bestMove iterative for differnt depths using AlphaBeta without the implementation of
                # transposition table
                __, bm = self.alpha_beta(board, depth, alpha, beta, max_player, color, 
                        time_limit)
    
            # If the time taken by the A.I is outside the specified time, return the current best move
            elapsed = time.time() - start
            print('Time Elapsed:{}'.format(elapsed))
            print('Time Limit:{}'.format(time_limit))
            if elapsed > time_limit:
            #if time.time() - start > time_limit or depth > maxDepth:
                print('searched till depth:{}'.format(depth))
                return bm
    
            # If the depth exceeds the board space, the bestMove is None. Thus we return the previous best value
            if bm is None:
                return prev_bm
    
            # Storing previous move for fallback procedure if the depth exceeds to the limits off the borad
            prev_bm = bm
            # incrementing depth and maxDepth with every iteration
            depth += 1
            maxDepth += 1
#------------------------------------------------------------------------------------------------
def weight(move,board):
    if board.is_empty(move):
        return 1
    else:
        return 0
#------------------------------------------------------------------------------------------------
def createGraph(board,color):
    
    active_nodes = [key for key in board.board.keys() if board.board[key] != board.get_opposite_color(color)]
    visited_nodes = [key for key in board.board.keys() if board.board[key] == board.get_opposite_color(color)]
    
    if color == board.RED:
        start = [node for node in active_nodes if node[1] == 0]
        end = [node for node in active_nodes if node[1] == board.size-1]
    else:
        start = [node for node in active_nodes if node[0] == 0]
        end = [node for node in active_nodes if node[0] == board.size-1]
    
    end = [node for node in active_nodes if board.border(color,node)]
    graph = dict()
    graph['start'] = {key:weight(key,board) for key in start}
    graph['end'] = {key:0 for key in end}
    
    for node in active_nodes:
        neighbors = board.get_neighbors(node)
        neighbors = list(filter(lambda x: (x not in visited_nodes), neighbors))
        link = {}
            
        for n in neighbors:
            link[n] = weight(n,board)
            
        graph[node] = link
        
    for node in start:
        graph[node]['start'] = weight(node,board)
    
    for node in end:
        graph[node]['end'] = 0
        
    return graph
#-----------------------------------------------------------------------------------------------------------
def dijkstra(graph, source, goal):  
    dist = {}
    prev = {}
    Q = set()
    Graph = graph.copy()
    # inf = 99
    
    # for each vortex v in Graph
    for v in Graph: 
        # dist[v] = inf
        dist[v] = np.Inf
        prev [v] = None
        Q.add(v) 
    dist[source] = 0
    #print("lenQ",len(Q))
    #print("lenGraph",len(Graph))
    
    while len(Graph) != 0: # Why does not work for len(Q)
        u = None
        for v in Graph:
            if u is None:
                u = v
            elif dist[v] < dist[u]:
                u = v
        
        for v, len_u_v in graph[u].items():
            alt = len_u_v + dist[u]
            if alt < dist[v]:
                dist[v] = alt
                
        # return last/most recent element of Graph
        Graph.pop(u)

    return dist[goal]
#-----------------------------------------------------------------------------
def makeMove(board, move, color):
    board.place(move, color)
    return board
#-----------------------------------------------------------------------------
def unMakeMove(board, move):
    board.board[move] = 3  # assigning empty board position
    if board.game_over == True:
        board.game_over = False
    return board
#-----------------------------------------------------------------------------
def getMoveList(board):
    # finding the co-ordinates which are empty given the board space
    lst = [key for key in board.board.keys() if board.board[key] == 3]
    return lst

trans_table = dict()

def lookup (board, depth):
    b = tuple(sorted(board.board.items()))
    #print(b)
    
    # find key 
    if (b, depth) in trans_table:
        g, ttbm = trans_table[(b, depth)]
        #print(g,ttbm)
        return True, g, ttbm
    
    else: # if trans_table empty 
        if not trans_table:
            return False, None, None

        # return bm given depth 
        keys = [key for key in trans_table.keys() if key[1] == depth]
        ttbm = None
        g = np.NINF
        # searhc trans_table for bm when no hit occured
        for key in keys:
            if trans_table[key][0] > g:
                g, ttbm = trans_table[key]
                # print(g,ttbm)
        return False, g, ttbm

def store(board, g, d, bm):
    # hash board state
    b = tuple(sorted(board.board.items()))
    global trans_table
    trans_table[(b, d)] = (g, bm) # store values of trans_table
    #print("Transposition Table",trans_table)