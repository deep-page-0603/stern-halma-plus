
from koni import *
from net import *
import numpy as np
import random

class Board(object):
	def __init__(self, drop_back = False):
		self.hop_tags = np.zeros(mini_board_size, dtype = np.int32)
		self.mini_board = np.zeros(mini_board_size, dtype = np.int8)
		self.info = np.zeros((2, 2), dtype = np.int8)
		
		self.leader = self.turn = None
		self.depth = 0
		self.drop_back = drop_back
		self.legals = []
		self.positions = [[], []]

	def reset(self, mini_board = boot_mini_board, turn = None, depth = 0, leader = None):
		self.info.fill(0)
		self.turn = turn if turn is not None else random.randint(0, 1)
		self.leader = leader if leader is not None else self.turn
		self.depth = depth
		self.mini_board[:] = mini_board[:]

		for t in range(2):
			ot = (t + 1) % 2
			self.positions[t] = [pt for pt in valid_mini_pts if mini_board[pt] == t]

			for pt in home_pts[t]:
				if mini_board[pt] == t:
					self.info[t, 0] += 1
				elif mini_board[pt] >= 0:
					self.info[ot, 1] += 1

		self._set_legals_()
		return self

	def go(self, act):
		t = self.turn
		ot = (t + 1) % 2

		if act is not None:
			spt, ept = act

			mb = self.mini_board
			mb[ept] = t
			mb[spt] = -1

			p = self.positions[t]
			sb, eb = boot_mini_board[spt], boot_mini_board[ept]

			p.remove(spt)
			p.append(ept)

			if sb != eb:
				if sb == t:
					self.info[t, 0] -= 1
				elif sb == ot:
					self.info[t, 1] -= 1

				if eb == t:
					self.info[t, 0] += 1
				elif eb == ot:
					self.info[t, 1] += 1
			
		self.turn = ot
		self.depth += 1
		self._set_legals_()

	def check_end(self, depth_out = False):
		rewards, locked_count = [0, 0], 0

		for t in range(2):
			ot = (t + 1) % 2

			if self.info[t, 1] > 0:
				if self.info[t, 1] == piece_count:
					if t == self.leader and self.info[ot, 1] == piece_count - 1:
						for spt, ept in self.legals:
							if boot_mini_board[spt] != t and boot_mini_board[ept] == t: return True, rewards, 0

					rewards[t], rewards[ot] = 1, -1
					return True, rewards, 0
				
				empty_count = piece_count - self.info[t, 1] - self.info[ot, 0]

				if empty_count == 0:
					rewards[t], rewards[ot] = 1, -1
					return True, rewards, 1

				if self.info[t, 0] + self.info[t, 1] == piece_count: locked_count += 1

		if locked_count == 2: return True, rewards, 1
		if depth_out and self.depth >= max_game_depth: return True, rewards, 2

		return False, None, None

	def evaluate(self, net, flipped = None):
		t = self.turn
		mb = self.mini_board if t == 0 else rotate(self.mini_board)

		if flipped is None: flipped = (random.randint(0, 1) == 0)
		fmb = mb[flip_mini_board] if flipped else mb

		state = np.zeros((3, mini_board_size), dtype = np.int8)
		state[0, fmb == 0] = 1
		state[1, fmb == 1] = 1
		if t == self.leader: state[2, :] = 1

		batch_pi, batch_v = net.eval([state])
		full_pi, v = batch_pi[0], batch_v[0]
		pi = []

		for spt, ept in self.legals:
			if t == 1:
				spt = mini_board_size - 1 - spt
				ept = mini_board_size - 1 - ept

			if flipped:
				spt = flip_mini_board[spt]
				ept = flip_mini_board[ept]

			pi.append(full_pi[spt, ept])

		if len(pi) > 0:
			return self.legals, np.array(pi) / (np.sum(pi) + 1e-10), v
		else:
			return [], None, v

	def get_state_pi(self, pi):
		t = self.turn
		mb = self.mini_board if t == 0 else rotate(self.mini_board)

		state = np.zeros((3, mini_board_size), dtype = np.int8)
		state[0, mb == 0] = 1
		state[1, mb == 1] = 1
		if t == self.leader: state[2, :] = 1

		pi_info = []

		for i, act in enumerate(self.legals):
			p = pi[i]
			if p < 1e-7: continue

			spt, ept = act

			if t == 1:
				spt = mini_board_size - 1 - spt
				ept = mini_board_size - 1 - ept

			pi_info.append((spt, ept, p))

		return state, pi_info

	def _set_legals_(self):
		t, mb = self.turn, self.mini_board
		ht, legals = self.hop_tags, self.legals

		legals.clear()
		ht.fill(-1)

		for pt in self.positions[t]:
			_walks_ = av_walks[pt]

			for i in range(av_walks_counts[pt]):
				wpt = _walks_[i]			
				if mb[wpt] < 0: legals.append((pt, wpt))

			hop_roots = [pt]
			ht[pt] = pt

			while len(hop_roots) > 0:
				new_roots = []

				for root in hop_roots:
					_walks_, _hops_ = hop_walks[root], hop_hops[root]

					for i in range(hops_counts[root]):
						wpt, hpt = _walks_[i], _hops_[i]

						if ht[hpt] != pt and mb[wpt] >= 0 and mb[hpt] < 0:
							ht[hpt] = pt
							new_roots.append(hpt)
							legals.append((pt, hpt))

				hop_roots = new_roots

		if self.drop_back:
			vb = value_mini_board[t]
			self.legals = [(spt, ept) for spt, ept in legals if vb[spt] <= vb[ept]]

if __name__ == '__main__':
	net = Net('./logs/best.ckpt')
	_, _, v = Board().reset(boot_mini_board).evaluate(net)

	print()
	print('boot_value = {:.5f}'.format(v))

	rb = get_random_mini_board()

	_, _, v = Board().reset(rb).evaluate(net, False)
	print('random_value = {:.5f}'.format(v))

	_, _, v = Board().reset(rb).evaluate(net, True)
	print('random_value (flipped) = {:.5f}'.format(v))
