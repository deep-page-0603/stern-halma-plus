
from koni import *
from util import *
from board import *
from net import *
from players import *
from mcts import *
import numpy as np
import time
import argparse

def test_koni():
	input('\n- press Enter to check boot board')
	click_board(boot_mini_board)

	input('\n- press Enter to check vacant mini-board')
	print_board(vacant_mini_board)

	input('\n- press Enter to check random mini-board')
	rand_board = get_random_mini_board()
	click_board(rand_board)

	input('\n- press Enter to check rotated random mini-board')
	click_board(rotate(rand_board))

	input('\n- press Enter to check flipped random mini-board')
	click_board(rand_board[flip_mini_board])

	input('\n- press Enter to check value mini-board(0)')
	print_board(value_mini_board[0], True)

	input('\n- press Enter to check value mini-board(1)')
	print_board(value_mini_board[1], True)

def test_random_player(verbose):
	player = RandomPlayer()
	test_players([player, player], drop_back = True, verbose = verbose)

def test_fast_player(model_path, verbose):
	net = Net(model_path)
	player = FastPlayer(net, random_start = False)
	test_players([player, player], drop_back = True, verbose = verbose)

def test_mcts_player(model_path, verbose):
	net = Net(model_path)
	player = MCTSPlayer(net, num_playouts, p_uct, 'valid', random_start = False, verbose = verbose)
	test_players([player, player], drop_back = True, verbose = verbose)

def test_comp(model_path_1, model_path_2):
	net_1 = Net(model_path_1)
	net_2 = Net(model_path_2)

	player_1 = MCTSPlayer(net_1, num_playouts, p_uct, 'valid')
	player_2 = MCTSPlayer(net_2, num_playouts, p_uct, 'valid')

	test_players([player_1, player_2], drop_back = True, verbose = True)

def test_players(players, drop_back, verbose):	
	board = Board(drop_back).reset()
	st = time.time()

	while True:
		end, rewards, exit = board.check_end(depth_out = True)
		if end: break

		act, _ = players[board.turn].think(board)
		board.go(act)

		if verbose: click_board(board.mini_board, act)

	dur = time.time() - st
	print(rewards, exit, board.depth, dur)

def test_saved(path):
	a = pickle_load(path)
	t = 0

	for s, _, _ in a:
		b = vacant_mini_board.copy()

		b[s[0] == 1] = 0
		b[s[1] == 1] = 1

		if t == 1: b = rotate(b)
		t = (t + 1) % 2

		click_board(b)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--case', type = str, default = 'koni', help = 'test case')
	args = parser.parse_args()

	if args.case == 'koni':
		test_koni()
	elif args.case == 'random':
		test_random_player(True)
	elif args.case == 'fast':
		test_fast_player('./logs/best.ckpt', True)
	elif args.case == 'mcts':
		test_mcts_player('./logs/best.ckpt', True)
	elif args.case == 'comp':
		test_comp('./logs/best.ckpt', './logs/best.ckpt')
	elif args.case == 'saved':
		test_saved('d.dat')
