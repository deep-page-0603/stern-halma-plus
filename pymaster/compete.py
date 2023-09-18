
from config import *
from koni import *
from util import *
from net import *
from players import *
from record import *
from board import *
import argparse
import gc

def compete(players, target, drop_back = False, verbose = False):
	records = [Record() for _ in range(2)]
	board = Board(drop_back = drop_back)
	
	total_prog = tqdm(range(target), desc = 'competing', leave = False) if target > 1 else None
	games, hashes = 0, []

	while True:
		gc.collect()
		
		board.reset()
		history = []

		for _ in tqdm(range(max_game_depth + 1), desc = 'game', leave = False):
			end, rewards, exit = board.check_end(depth_out = True)
			if end: break

			act, _ = players[board.turn].think(board)
			
			board.go(act)
			if act is not None: history.append(act)

			if verbose: click_board(board.mini_board, act)

		h = hash(tuple(history))
		
		if h in hashes: continue
		if total_prog: total_prog.update(1)

		hashes.append(h)

		for t in range(2):
			records[t].add(rewards[t], exit, board.depth)

		games += 1
		if games == target: break

		if verbose:
			print()
			print(str(records[0]))
			print()

	if total_prog: total_prog.close()
	return records

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-a', '--agent', type = str, default = './logs/best.ckpt', help = 'agent model path')
	parser.add_argument('-e', '--env', type = str, default = './logs/best.ckpt', help = 'env model path')
	parser.add_argument('-g', '--games', type = int, default = valid_games, help = 'number of games')
	parser.add_argument('-d', '--drop', action = 'store_true', default = False, help = 'drop backward acts')
	parser.add_argument('-v', '--verbose', action = 'store_true', default = False, help = 'verbose')
	args = parser.parse_args()

	net_1 = Net(get_model_path(args.agent))
	net_2 = Net(get_model_path(args.env))

	player_1 = MCTSPlayer(net_1, num_playouts, p_uct, 'valid')
	player_2 = MCTSPlayer(net_2, num_playouts, p_uct, 'valid')

	print()
	record = compete([player_1, player_2], args.games, args.drop, args.verbose)[0]

	print()
	print(str(record))
