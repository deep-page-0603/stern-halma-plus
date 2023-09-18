
from config import *
from util import *
from koni import *
from board import *
from players import *
from net import *
from tqdm import tqdm
import numpy as np
import random
import time
import argparse
import gc

def selfplay(player, drop_back, verbose):
	board = Board(drop_back = drop_back).reset()
	s_pi_t = []
	gc.collect()

	for _ in tqdm(range(max_game_depth + 1), desc = 'game', leave = False):
		end, rewards, exit = board.check_end(depth_out = True)
		if end: break

		if board.depth == 0:
			act, info = player.think(board)
		else:
			act, info = player.think_again(board, act)

		s_pi_t.append(info)
		board.go(act)

		if verbose: click_board(board.mini_board, act)

	s_pi_v = [(s, pi_info, -rewards[t]) for s, pi_info, t in s_pi_t]
	return s_pi_v, exit

def write_to_file(dst_dir, s_pi_v, exit):
	path = '{}/{}_{:06}_{}_{}.dat'.format(
		dst_dir,
		time.strftime('%Y%m%d%H%M%S', time.localtime()),
		random.randint(1, 999999),
		exit,
		len(s_pi_v)
	)
	pickle_dump(s_pi_v, path)

def main(args):
	net = Net('./logs/best.ckpt')	
	player = MCTSPlayer(net = net, playouts = num_playouts, p_factor = p_uct, mode = 'train', random_start = True)
	
	print()
	prog_bar = tqdm(float('inf'), desc = 'total')

	if args.url == 'localhost':
		dst_dir = './data/stage_{}'.format(args.stage)
		mkdir(dst_dir)
	else:
		dst_dir = '\\\\{}\\data\\stage_{}'.format(args.url, args.stage)
	
	while True:
		s_pi_v, exit = selfplay(player, args.drop, args.verbose)
		if exit == 2: continue

		write_to_file(dst_dir, s_pi_v, exit)
		prog_bar.update(1)

	prog_bar.close()

if __name__ == '__main__':	
	parser = argparse.ArgumentParser()
	parser.add_argument('-u', '--url', type = str, default = '192.168.1.150', help = 'train server url')
	parser.add_argument('-s', '--stage', type = int, default = '1', help = 'stage')
	parser.add_argument('-d', '--drop', action = 'store_true', default = False, help = 'drop backward acts')
	parser.add_argument('-v', '--verbose', action = 'store_true', default = False, help = 'verbose')
	args = parser.parse_args()

	main(args)
