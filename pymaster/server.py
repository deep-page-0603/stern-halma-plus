
from flask import Flask
from config import *
from util import *
from koni import *
from board import *
from net import *
from players import *
import numpy as np
import random
import argparse

app = Flask(__name__)

@app.route('/')
def index():
    return 'welcome to duel-pykoni server!'

@app.route('/sos/<data>', methods = ['POST'])
def sos(data):
	global red_player, blue_player, drop_back

	segs = data.split(':')
	turn = int(segs[0])
	depth = int(segs[1])
	leader = int(segs[2])	
	mini_board = translate_board(segs[3])
	
	board = Board(drop_back = drop_back).reset(mini_board, turn, depth, leader)
	player = red_player if turn == 0 else blue_player

	act, _ = player.think(board)
	spt, ept = act

	resp = str(spt | (ept << 8))
	resp = '375077428868303736' + resp
	return resp

def translate_board(mat_str):
	board = np.array([int(x) for x in mat_str.split(',')]).astype(np.int8)
	return board

def main(args):
	global app, red_player, blue_player, drop_back

	red_net = Net(get_model_path(args.red))
	blue_net = Net(get_model_path(args.blue))

	type_segs = args.type.split(',')
	red_segs = type_segs[0].split(':')
	blue_segs = type_segs[1].split(':')

	drop_back = args.drop

	if red_segs[0] == 'f':
		red_player = FastPlayer(net = red_net, random_start = (red_segs[1] == '1'))
	elif red_segs[0] == 'm':
		red_player = MCTSPlayer(net = red_net, playouts = int(red_segs[2]), p_factor = float(red_segs[3]),
			mode = 'valid', random_start = (red_segs[1] == '1'), verbose = True)

	if blue_segs[0] == 'f':
		blue_player = FastPlayer(net = blue_net, random_start = (blue_segs[1] == '1'))
	elif blue_segs[0] == 'm':
		blue_player = MCTSPlayer(net = blue_net, playouts = int(blue_segs[2]), p_factor = float(blue_segs[3]),
			mode = 'valid', random_start = (blue_segs[1] == '1'), verbose = True)

	segs = args.url.split(':')
	app.run(host = segs[0], port = segs[1], debug = False)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-u', '--url', type = str, default = '127.0.0.1:1989', help = 'server url')
	parser.add_argument('-r', '--red', type = str, default = './logs/best.ckpt', help = 'red player model path')
	parser.add_argument('-b', '--blue', type = str, default = './logs/best.ckpt', help = 'blue player model path')
	parser.add_argument('-t', '--type', type = str, default = 'm:0:400:3.0,m:0:400:3.0', help = 'player types')
	parser.add_argument('-d', '--drop', action = 'store_true', default = False, help = 'drop backward acts')
	args = parser.parse_args()

	main(args)
