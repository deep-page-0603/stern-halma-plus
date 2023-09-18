
from config import *
from util import *
from koni import *
from mcts import *
from tqdm import tqdm
import numpy as np
import random

class Player(object):
	def __init__(self, mode):
		self.mode = mode

		self.start_legals = [
			[(16, 27), (26, 37), (36, 37), (36, 47), (46, 47), (56, 57)],
			[(104, 93), (94, 83), (84, 83), (84, 73), (74, 73), (64, 63)]
		]

	def _think_(self, pi, depth):
		if len(pi) == 0: return None

		if self.mode == 'valid' or depth > warm_depth:
			pi = softmax(pi / cold_temper)
		else:
			pi = softmax(pi)

		return np.random.choice(len(pi), p = pi)

class RandomPlayer(Player):
	def __init__(self):
		super(RandomPlayer, self).__init__('valid')

	def think(self, board):		
		if len(board.legals) > 0:
			return random.sample(board.legals, 1)[0], None
		else:
			return None, None

class FastPlayer(Player):
	def __init__(self, net, random_start = False):
		super(FastPlayer, self).__init__('valid')
		
		self.net = net
		self.random_start = random_start

	def think(self, board):
		if self.random_start and board.depth < 2:
			return random.sample(self.start_legals[board.turn], 1)[0], None

		legals, pi, _ = board.evaluate(self.net)
		idx = self._think_(pi, board.depth)

		if idx is not None:
			return legals[idx], None
		else:
			return None, None

class MCTSPlayer(Player):
	def __init__(self, net, playouts, p_factor, mode, random_start = False, verbose = False):
		super(MCTSPlayer, self).__init__(mode)

		self.mcts = MCTS(net, playouts, p_factor, verbose)
		self.random_start = random_start
		self.is_train = (self.mode == 'train')

	def think(self, board):
		self.root = self.mcts.run(board, None, self.is_train)
		return self._inner_think_(board)

	def think_again(self, board, act):
		self.root = self.root.children[act]
		self.mcts.run(board, self.root, self.is_train)
		return self._inner_think_(board)

	def _inner_think_(self, board):
		legals, visits = [], []

		for act, child in self.root.children.items():
			legals.append(act)
			visits.append(child.visits)

		pi = np.array(visits, dtype = np.float32)
		idx = self._think_(pi, board.depth)
		
		act = legals[idx]
		pi = pi / np.sum(pi)

		if self.random_start and board.depth < 2:
			act = random.sample(self.start_legals[board.turn], 1)[0]

		if self.is_train:
			state, pi_info = board.get_state_pi(pi)
			return act, (state, pi_info, board.turn)
		else:
			return act, None
