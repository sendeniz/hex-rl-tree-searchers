import random
import numpy as np
import time
from hex_skeleton import HexBoard
import copy

run_time = 0 
elapsed_time = 0
beta_cutoff = 0
alpha_cutoff = 0

tt_beta_cutoff = 0
tt_alpha_cutoff = 0

maxDepth = 1
AI_color = HexBoard.BLUE

class Searcher:

    def __init__(self, depth, policy, board, color, eval = None, time_limit = 1, 
                 use_tt = False, max_iter = 100, C = np.sqrt(2)):
        
        self.depth = depth
        self.policy = policy
        self.board = board
        self.color = color
        self.eval = eval
        self.time_limit = time_limit
        self.use_tt = use_tt
        self.max_iter = max_iter
        self.C = C
    
    def find_bm(self):
        
        # agent will use random action (dummy evaluation) policy
        if self.policy == "random":
            moveList = getMoveList(self.board)
            return random.choice(moveList)
        
        # agent will use alpha beta as policy
        elif self.policy == "alphabeta":
            __, bm = self.alpha_beta(board = self.board, depth = self.depth, 
                                     alpha = np.NINF, beta = np.Inf, max_player = True, 
                                     color = self.color, 
                                     time_limit = self.time_limit)
            return bm
        
        # agent will use alpha beta policy with transposition tables
        elif self.policy == "ttalphabeta":
            __, bm = self.tt_alpha_beta(board = self.board, depth = self.depth, 
                                        alpha = np.NINF, beta = np.Inf, max_player = True, 
                                        color = self.color,
                                        time_limit = self.time_limit)
            return bm
        
        # agent will use alpha beta policy with iterative deepning with or without 
        # transposition tables
        elif self.policy == "iterdeep":
            bm = self.iter_deep(board = self.board,
                                        alpha = np.NINF, beta = np.Inf, max_player = True, 
                                        color = self.color,
                                        time_limit = self.time_limit, use_tt=self.use_tt)
            return bm
        
        # agent will use monte carlo tree search policy 
        elif self.policy == 'mcts':
            bm = self.mc_tree_search(board = self.board, color = self.color, 
                           max_iter = self.max_iter, C = self.C)
            return bm
                
    def heuristic_eval(self,board):
        agent_color = self.color
        opposite_color = board.get_opposite_color(agent_color)
        
        dijk_graph_1 = makeGraph(board,agent_color)
        value_1 = dijkstra(dijk_graph_1, 'start', 'end')
        
        dijk_graph_2 = makeGraph(board, opposite_color)
        value_2 = dijkstra(dijk_graph_2, 'start', 'end')
        
        # intead of simple dijkstra dist compute objective dijkstra dist
        # as difference between both player values
        obj_d_dist = value_2 - value_1
        return obj_d_dist, (10,10)
    
    def alpha_beta(self, board, depth, alpha, beta, max_player, color, 
                   time_limit):   
        # print("Alpha Beta Policy")
        global alpha_cutoff
        global beta_cutoff
        global run_time
        global elapsed_time 
        
        start_time = time.time()
        bm = None # initalize best moves to store
        
        moveList = getMoveList(board)
        if (depth <= 0) or (board.game_over):
            # board.print()
            return self.heuristic_eval(board)
        
        elif max_player: # Max Player
            
            g = np.NINF # best value g
            
            # iterate over c moves 
            for move in moveList:
                board = makeMove(board, move, color)
                
                gc, __ = self.alpha_beta(board, depth - 1, alpha, beta, 
                                         False, time_limit, 
                                         board.get_opposite_color(color))
                board = unMakeMove(board, move)
                
                if gc >= g : # if value gc  >= best value g
                    bm = move # save best move information
                g = max(gc, g) # max (value, best value)
                alpha = max(alpha, g) # update alpha
                #print("alpha:({}),beta:({}),value gc:({}), value g:({})".format(alpha , beta, gc, g))


                if alpha >= beta:
                    # global beta_cutoff
                    beta_cutoff +=1
                    # print("beta cutoff:({})".format(beta_cutoff))
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
                
                if gc <= g: # if value gc <= best value g
                    bm = move # store best move info
                
                g = min(gc, g) # min(value, best value)
                beta = min(beta, g) # update beta
                #print("alpha:({}),beta:({}),value gc:({}), value g:({})".format(alpha , beta, gc, g))

                if alpha >= beta:
                    global alpha_cutoff
                    alpha_cutoff +=1
                    # print("alpha cutoff:({})".format(alpha_cutoff))
                    break
                
            run_time += time.time() - start_time
            # print("Sys Run Time:({})".format(run_time))
            # board.print()
        return g, bm # return best value and best move
        
    def tt_alpha_beta(self, board, depth, alpha, beta, max_player, color,
                      time_limit):   
        # print("TT Alpha Beta Active")
        global run_time
        global tt_beta_cutoff
        global tt_alpha_cutoff
        
        start_time = time.time()
        
        delta_depth = maxDepth - depth
        (hit, g, ttbm) = lookup(board, delta_depth)
        # print((hit, g, ttbm))
        
        # if hit occurs then and if field is empty then:
        if hit and delta_depth >= 2 and board.board[ttbm] == 3:
            return g, ttbm
            
        bm = None # initalize best moves to store
        
        # if no hit and transposition table is empty then:
        if ttbm is None:
            moveList = getMoveList(board)
        
        # if not hit and transposition table is not empty then:
        else:
            moveList = [ttbm] + getMoveList(board)
        
        if (depth <= 0) or (board.game_over):
            return self.heuristic_eval(board), bm
        
        elif max_player: # Max Player
            
            g = np.NINF # best value g
            
            # iterate over c moves to find best move
            for move in moveList: 
                board = makeMove(board, move, color)
                
                gc, __ = self.tt_alpha_beta(board, depth-1, alpha, beta, 
                                            False, board.get_opposite_color(color),
                                            time_limit)
                
                if isinstance (gc, tuple) == True:
                    gc = gc[0]
                else :
                    gc = gc
                
                board = unMakeMove(board,move)
               
                if gc >= g : # if value gc  >= best value g
                    bm = move # save best move information
                    g = gc
                
                alpha = max(alpha,g) # update alpha
                # print("alpha:({}),beta:({}),value gc:({}), value g:({})".format(alpha , beta, gc, g))
                
                if alpha >= beta:
                    tt_beta_cutoff +=1
                    # print("tt beta cutoff:({})".format(tt_beta_cutoff))
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
                    gc = gc[0]
                else :
                    gc = gc
                
                board = unMakeMove(board, move)
                
                if gc <= g :
                    bm = move # store best move info
                    g = gc
                
                beta = min(beta, g) # update beta
                # print("alpha:({}),beta:({}),value gc:({}), value g:({})".format(alpha , beta, gc, g))
                
                if alpha >= beta:
                    tt_alpha_cutoff +=1
                    # print("tt alpha cutoff:({})".format(tt_alpha_cutoff))
                    break
            
            run_time += time.time() - start_time
            # print("run_time:",run_time)
            # board.print()
        save(board, g, delta_depth, bm)
        return g, bm # return best value and best move
    
    def iter_deep(self, board, max_player, alpha, beta, color, time_limit, use_tt):
        global maxDepth

        # Start timer
        start = time.time()
        # initialize depth
        depth = 1
        # Initialize prevBestmove
        prev_bm = tuple()
    
        while True:
            if use_tt:
                
                # find the bm  with use of transposition tables and iterative depth
                __, bm = self.tt_alpha_beta(board, depth, alpha, beta, max_player, color,
                          time_limit)
    
            else:
                # find the bm  without use of transposition tables and iterative depth
                __, bm = self.alpha_beta(board, depth, alpha, beta, max_player, color, 
                        time_limit)
    
            elapsed = time.time() - start
            # print('Time Elapsed:{}'.format(elapsed))
            # print('Time Limit:{}'.format(time_limit))
            # print('Time is UP:{}'.format(elapsed >= time_limit))
            # if not time limit is left then:
            if elapsed >= time_limit:
            #if time.time() - start > time_limit or depth > maxDepth:
                # print('sarched depth:{}'.format(depth))
                return bm
    
            # if boardspace exceeds depth, then bm is none and return previously
            # store bm
            if bm is None:
                return prev_bm
    
            # fail save previous bm 
            prev_bm = bm

            depth += 1
            maxDepth += 1

    def mc_tree_search(self, board, color, max_iter, C):
        # print("Max iter set to{}:",max_iter)
        # print("C_val set to{}:",C)
        
        rootnode = Node(board, color)
        for _ in range(max_iter):
            node = rootnode
            # state s
            s = copy.deepcopy(rootnode.board)
    
            # select node
            while node.new_m == [] and node.childNodes != []:
                # print("Selection Active")
    
                node = node.ucts(C)
                # state s
                s = makeMove(s, node.move, s.get_opposite_color(node.Ncolor))
    
            # expand
            if node.new_m:
    
                mve = node.new_m[np.random.choice(len(node.new_m))]
                s = makeMove(s, mve, node.Ncolor)
                node.new_m.remove(mve)
                child = Node(s, s.get_opposite_color(node.Ncolor))
                child.move = mve
                
                if s.is_game_over:
                    child.terminal = True
                node.add_child(child)
                node = child
            
            # obtain move list and shuffle
            m = getMoveList(s)
            np.random.shuffle(m)
            
            # player 1 and player 2 moves
            p1_m = m[:len(m) // 2]
            p2_m = m[len(m) // 2:]
            s.board.update(zip(p1_m, [node.Ncolor] * len(p1_m)))
            s.board.update(zip(p2_m, [s.get_opposite_color(node.Ncolor)] * len(p2_m)))
    
            result = s.check_win(AI_color)
    
            s.board.update(zip(m, [HexBoard.EMPTY] * len(m)))
    
            
            # perform backpropagation
            while node is not None:
                node = node.update_node_params(int(result))
    
        # print('time-taken: {}'.format(time.time()-start))
        bm = sorted(rootnode.childNodes, key = lambda c: c.v)[-1].move
        return bm
#-----------------------------------------------------------------------------
def makeMove(board, move, color):
    board.place(move, color)
    return board

def unMakeMove(board, move):
    board.board[move] = 3  # assigning empty board position
    if board.game_over == True:
        board.game_over = False
    return board

def getMoveList(board):
    # finding tboard space x-y axis which are empty
    lst = [key for key in board.board.keys() if board.board[key] == 3]
    return lst

# obtain weights w 
def w(move, board):
    if board.is_empty(move):
        return 1
    else:
        return 0

def makeGraph(board, color):
    
    # obtain active nodes
    a_nodes = [key for key in board.board.keys() if board.board[key] != board.get_opposite_color(color)]
    # obtain visited nodes
    v_nodes = [key for key in board.board.keys() if board.board[key] == board.get_opposite_color(color)]
    
    if color == board.RED:
        start = [node for node in a_nodes if node[1] == 0]
        end = [node for node in a_nodes if node[1] == board.size - 1]
    
    else:
        start = [node for node in a_nodes if node[0] == 0]
        end = [node for node in a_nodes if node[0] == board.size - 1]
    
    end = [node for node in a_nodes if board.border(color,node)]
    graph = dict()
    graph['start'] = {key:w(key,board) for key in start}
    graph['end'] = {key:0 for key in end}
    
    for node in a_nodes:
        neighbors = board.get_neighbors(node)
        neighbors = list(filter(lambda x: (x not in v_nodes), neighbors))
        link = {}
            
        for n in neighbors:
            link[n] = w(n, board)
            
        graph[node] = link
        
    for node in start:
        graph[node]["start"] = w(node, board)
    
    for node in end:
        graph[node]["end"] = 0
        
    return graph

def dijkstra(graph, source, goal):  
    dist = {}
    prev = {}
    Q = set()
    Graph = graph.copy()
    
    # for each vortex v in Graph
    for v in Graph: 
        # dist[v] = inf
        dist[v] = np.Inf
        prev [v] = None
        Q.add(v) 
    dist[source] = 0
    
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

#-----Alpha Beta Table Enhancements-------------------------------------------
trans_table = dict()

def lookup (board, depth):
    b = tuple(sorted(board.board.items()))
    #print(b)
    
    # find key 
    if (b, depth) in trans_table:
        g, ttbm = trans_table[(b, depth)]
        # print(g,ttbm)
        return True, g, ttbm
    
    # if trans_table empty then:
    else: 
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

def save(board, g, d, bm):
    # hash board state
    b = tuple(sorted(board.board.items()))
    global trans_table
    trans_table[(b, d)] = (g, bm) # store values of trans_table
    # print("Transposition Table", trans_table)
    
#---Monte Carlo Tree Search---------------------------------------------------
class Node:

    def __init__(self, board, color):

        # node params
        self.v = 1 # visited node
        self.w = 0 # win node

        # tree params: parent N, child n
        self.parent = None
        self.childNodes = []

        # state s
        self.board = board
        self.Ncolor = color

        # terminal node/terminal leaf
        self.terminal = False

        # availaible moves as new moves that have not been tried before
        self.new_m = getMoveList(self.board)
        self.move = None

    def add_child(self, obj):
        # add child to parent node

        obj.parent = self
        self.childNodes.append(obj)

    def ucts(self, C = np.sqrt(2)):
        # float C is exploration/explotation tradeoff
        max_val = 0
        bestNode = None

        for child in self.childNodes:
            
            uct = (child.w / child.v) + C * (np.sqrt(np.log(child.parent.v) / child.v))

            if uct >= max_val:
                max_val = uct
                bestNode = child
        return bestNode

    def update_node_params(self, result):
        
        # v vistied, w win
        self.v += 1
        self.w += result

        return self.parent