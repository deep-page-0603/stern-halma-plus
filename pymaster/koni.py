
from config import *
import numpy as np
import random

vacant_mini_board = np.zeros(mini_board_size, dtype = np.int8)
boot_mini_board = np.zeros(mini_board_size, dtype = np.int8)
value_mini_board = np.zeros((2, mini_board_size), dtype = np.int8)
flip_mini_board = np.zeros(mini_board_size, dtype = np.int8)

valid_mini_pts = []
home_pts = [[], []]

av_walks = np.zeros((mini_board_size, 6), dtype = np.int8)
av_walks_counts = -np.ones(mini_board_size, dtype = np.int8)

hop_walks = np.zeros((mini_board_size, 6), dtype = np.int8)
hop_hops = np.zeros((mini_board_size, 6), dtype = np.int8)
hops_counts = -np.ones(mini_board_size, dtype = np.int8)

is_inited = False

def init_koni():
	global is_inited

	if not is_inited:
		vacant_2d_board = _get_vacant_2d_board_()
		boot_2d_board = _get_boot_2d_board_(vacant_2d_board)
		value_2d_board = _get_value_2d_board_(vacant_2d_board)

		vacant_mini_board[:] = _2d_board_to_mini_board_(vacant_2d_board)[:]
		boot_mini_board[:] = _2d_board_to_mini_board_(boot_2d_board)[:]
		
		value_mini_board[0, :] = _2d_board_to_mini_board_(value_2d_board)[:]
		value_mini_board[1, :] = value_mini_board[0, ::-1]

		_set_valid_mini_pts_()
		_set_home_pts_()	
		_set_av_walks_()
		_set_hops_()
		_set_flip_mini_board_()

		print('\n* duel-pykoni initialized with dim={} and valid_pts={}'.format(dim, len(valid_mini_pts)))
		is_inited = True

def rotate(mini_board):
	board = mini_board.copy()
	board[mini_board == 0] = 1
	board[mini_board == 1] = 0
	return board[::-1]

def get_random_mini_board():
	board = vacant_mini_board.copy()
	selected_pts = random.sample(valid_mini_pts, 2 * piece_count)

	for i in range(piece_count):
		board[selected_pts[i]] = 0
		board[selected_pts[i + piece_count]] = 1

	return board

def pt_to_mini_pt(pt):
	x, y = pt // board_dim, pt % board_dim
	x -= dim_1 - 1
	y -= dim_1 - 1
	return _mini_pt_(x, y)

def _get_vacant_2d_board_():
	board = -2 * np.ones((board_dim, board_dim), dtype = np.int8)

	for x in range(tri_dim + 1):
		for y in range(tri_dim + 1):
			if y >= tri_dim - x: board[x, y] = -1
			if y <= tri_dim - x: board[dim_1 + x, dim_1 + y] = -1

	return board

def _get_boot_2d_board_(vacant_2d_board):
	board = vacant_2d_board.copy()

	for x in range(dim):
		for y in range(dim):
			if y <= dim_1 - x: board[dim_1 + x, dim_1 + y] = 0
			if y >= dim_1 - x: board[dbl_dim + x, dbl_dim + y] = 1

	return board

def _get_value_2d_board_(vacant_2d_board):
	board = -np.ones((board_dim, board_dim), dtype = np.int8)

	for x in range(board_dim):
		for y in range(board_dim):
			if vacant_2d_board[x, y] == -2: continue

			v = x + y - dbl_dim

			if min(x, y) < dim_1:
				v -= dim_1 - min(x, y)
			elif max(x, y) > tri_dim:
				v -= max(x, y) - tri_dim

			board[x, y] = v

	return board

def _set_valid_mini_pts_():
	valid_mini_pts.clear()
	valid_mini_pts.extend([pt for pt in range(mini_board_size) if vacant_mini_board[pt] == -1])

def _set_home_pts_():
	for pt in valid_mini_pts:
		if boot_mini_board[pt] == 0:
			home_pts[0].append(pt)
		elif boot_mini_board[pt] == 1:
			home_pts[1].append(pt)

def _set_flip_mini_board_():
	board = np.arange(mini_board_size, dtype = np.int8)
	board = board.reshape((mini_board_dim, mini_board_dim)).transpose()
	board = board.reshape((-1))
	flip_mini_board[:] = board

def _set_av_walks_():
	deltas = [(0, 1), (1, 0), (1, -1), (-1, 1), (0, -1), (-1, 0)]

	for pt in valid_mini_pts:
		x, y = _parse_mini_pt_(pt)
		count = 0

		for dx, dy in deltas:
			wx, wy = x + dx, y + dy
			if not _is_in_mini_board_(wx, wy): continue

			wpt = _mini_pt_(wx, wy)
			if vacant_mini_board[wpt] == -2: continue

			av_walks[pt, count] = wpt
			count += 1

		av_walks_counts[pt] = count

def _set_hops_():
	deltas = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, 1)]

	for pt in valid_mini_pts:
		x, y = _parse_mini_pt_(pt)
		count = 0

		for dx, dy in deltas:
			wx, wy = x + dx, y + dy
			hx, hy = x + 2 * dx, y + 2 * dy

			if not _is_in_mini_board_(hx, hy): continue

			wpt = _mini_pt_(wx, wy)
			hpt = _mini_pt_(hx, hy)

			if vacant_mini_board[hpt] == -2: continue

			hop_walks[pt, count] = wpt
			hop_hops[pt, count] = hpt
			count += 1

		hops_counts[pt] = count

def _2d_board_to_mini_board_(board):
	return board[dim_1 - 1:tri_dim + 2, dim_1 - 1:tri_dim + 2].reshape([-1])

def _mini_board_to_2d_board_(board):
	d_board = _get_vacant_2d_board_()
	d_board[dim_1 - 1:tri_dim + 2, dim_1 - 1:tri_dim + 2] = board.reshape([mini_board_dim, mini_board_dim])
	return d_board

def _mini_pt_(x, y):
	return y + x * mini_board_dim

def _parse_mini_pt_(pt):
	return pt // mini_board_dim, pt % mini_board_dim

def _is_in_mini_board_(x, y):
	return x >= 0 and y >= 0 and x < mini_board_dim and y < mini_board_dim

init_koni()
