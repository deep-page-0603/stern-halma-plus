
from tqdm import tqdm
from config import *
from util import *
from koni import *
from net import *
import numpy as np
import random
import copy

class Node(object):
	def __init__(self, parent = None, p = 0):
		self.parent = parent
		self.children = {}
		self.visits = 0
		self.p = p
		self.q = self.w = 0

	def update(self, leaf_v):
		self.visits += 1
		self.w += leaf_v
		self.q = self.w / self.visits

		if self.parent: self.parent.update(-leaf_v)

	def expand(self, legals, pi):
		if len(legals) > 0:
			for i, act in enumerate(legals):
				self.children[act] = Node(self, pi[i])
		else:
			self.children[None] = Node(self, 0)

	def add_noise(self):
		noise = noise_factor * np.random.dirichlet(dirichlet_alpha * np.ones(len(self.children)))
		i = 0

		for _, child in self.children.items():
			child.p = (1 - noise_factor) * child.p + noise[i]
			i += 1

	def is_leaf(self):
		return len(self.children) == 0

class MCTS(object):
	def __init__(self, net, playouts, p_factor, verbose = False):
		self.net = net
		self.playouts = playouts
		self.p_factor = p_factor
		self.verbose = verbose

	def run(self, board, root, noisy):
		if root is None: root = Node()

		ran = range(self.playouts)
		if self.verbose: ran = tqdm(ran, desc = 'mcts')

		noise_added = False

		for _ in ran:
			if noisy and not noise_added and not root.is_leaf():
				root.add_noise()
				noise_added = True

			self._playout_(copy.deepcopy(board), root)

		return root

	def _playout_(self, board, root):
		node = root

		while not node.is_leaf():
			act = self._select_(node)
			node = node.children[act]
			board.go(act)

		end, rewards, _ = board.check_end()

		if not end:
			legals, pi, leaf_v = board.evaluate(self.net)
			node.expand(legals, pi)
		else:
			leaf_v = -rewards[board.turn]

		node.update(leaf_v)

	def _select_(self, node):
		parent_factor = self.p_factor * np.sqrt(node.visits)
		best_act, max_uct = None, -1e10

		for act, child in node.children.items():
			uct = child.q + parent_factor * child.p / (1 + child.visits)

			if uct > max_uct:
				max_uct = uct
				best_act = act

		return best_act
