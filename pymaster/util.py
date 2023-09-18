
from config import *
from koni import *
import numpy as np
import pickle
import shutil
import click
import glob
import os

def mkdir(path):
	os.makedirs(path, exist_ok = True)

def rmdir(path, remove_self = False):
	gl = sorted(glob.glob(path + '/**', recursive = True))

	for g in gl:
		if not os.path.isdir(g): os.remove(g)

	gl = gl[::-1] if remove_self else gl[:1:-1]

	for g in gl:
		if os.path.isdir(g): os.rmdir(g)

def backup_src(path):
	mkdir(path)

	for pyfile in os.listdir('./'):
		if not pyfile.endswith('.py'): continue
		if os.path.exists(path + '/' + pyfile): continue

		shutil.copy(pyfile, path + '/' + pyfile)

def softmax(x):
	probs = np.exp(x - np.max(x))
	probs /= np.sum(probs)
	return probs
	
def pickle_dump(data, path):
	with open(path, 'wb') as f:
		pickle.dump(data, f)

def pickle_load(path):
	with open(path, 'rb') as f:
		return pickle.load(f)

def click_board(board, act = None, indent = 2):
	print()

	spt, ept = act if act is not None else (-1, -1)
	pt = 0

	for x in range(mini_board_dim):
		l = click.style(''.join(['\t' for _ in range(indent)]))

		for y in range(mini_board_dim):
			v = board[pt]
			s, fg, bg = ' ', 'reset', 'reset'

			if v == -2:
				s = ' '
			elif v == -1:
				s = '*'
				if spt == pt: fg = 'yellow'
			else:
				s = chr(65 + v)

				if ept == pt:
					fg = 'bright_red' if v == 0 else 'bright_cyan'
				else:
					fg = 'red' if v == 0 else 'cyan'

			l += click.style(s + ' ', fg = fg, bg = bg)
			pt += 1

		click.echo(l)

	print()

def print_board(board, is_scalar_board = False, indent = 2, fp = None):
	def write(msg = ''):
		if fp:
			fp.write(msg + '\n')
		else:
			print(msg)

	write()

	if len(board.shape) == 2:
		d = board.shape[0]
		board = board.reshape([-1])
	elif len(board.shape) == 1:
		d = int(np.sqrt(board.shape[0]))
	else:
		write('### only 1d or 2d boards can be printed, but got {}d.\n'.format(len(board.shape)))
		return

	pt = 0

	for x in range(d):
		l = ''.join(['\t' for _ in range(indent)])

		for y in range(d):
			v = board[pt]
			pt += 1
			l += ptv_to_str(v, is_scalar_board) + ' '

		write(l)

	write()

def print_1darray(arr):
	print('[' + ','.join(['{:.2f}'.format(x) for x in arr]) + ']')

def ptv_to_str(v, is_scalar_board = False):
	if not is_scalar_board:
		if v == -2: return ' '
		if v == -1: return '*'
		return chr(65 + v)
	else:
		if v < 0: return '  '
		return '{:2}'.format(v)

def bilog(path, msg = ''):
	with open(path, 'a') as f:
		f.write(msg + '\n')

	print(msg)

def load_history(path):
	with open(path, 'rb') as fp:
		l = []

		while True:
			x = np.fromfile(fp, dtype = np.int32, count = 1)
			if len(x) == 0: break

			x = x[0]
			spt, ept = x & 0xFFFF, x >> 16

			spt = pt_to_mini_pt(spt)
			ept = pt_to_mini_pt(ept)

			l.append((spt, ept))

		return l

def get_model_path(fname):
	if os.path.exists(fname): return fname
	paths = glob.glob('./logs/**', recursive = True)

	for f in paths:
		if fname in f: return f

	return fname
