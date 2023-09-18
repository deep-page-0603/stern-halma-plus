
from tqdm import tqdm
from config import *
from koni import *
from util import *
from board import *
import numpy as np
import random
import glob
import os

def read_from_file(path):
	board = Board(drop_back = True)

	with open(path, 'rb') as fp:
		data = []

		while True:
			b = np.fromfile(fp, dtype = np.int8, count = mini_board_size)
			if len(b) == 0: break

			sb = np.fromfile(fp, dtype = np.int8, count = mini_board_size)
			eb = np.fromfile(fp, dtype = np.int8, count = mini_board_size)
			spi = np.fromfile(fp, dtype = np.float32, count = mini_board_size)
			epi = np.fromfile(fp, dtype = np.float32, count = mini_board_size)
			v = np.fromfile(fp, dtype = np.float32, count = 1)[0]
			t = np.fromfile(fp, dtype = np.int8, count = 1)[0]

			if v == 0: continue

			if t == 1:
				b = rotate(b)
				spi = spi[::-1]
				epi = epi[::-1]
			
			board.reset(b, 0)
			pi_info = []

			for spt, ept in board.legals:
				x = spi[spt] * epi[ept]
				if x < 1e-7: continue

				pi_info.append((spt, ept, x))

			s = np.zeros((3, mini_board_size), dtype = np.int8)
			s[0, b == 0] = 1
			s[1, b == 1] = 1
			if random.randint(0, 1) == 0: s[2, :] = 1

			data.append((s, pi_info, v))

		return data

src_dir = 'e:/koni/pymaster/duel/games'
mkdir('./data/stage_0')
i = 0

for file in tqdm(glob.glob(src_dir + '/**', recursive = True), desc = 'loading'):
	if os.path.isdir(file): continue
	if not file.endswith('.dat'): continue

	data = read_from_file(file)
	pickle_dump(data, './data/stage_0/{:06}.dat'.format(i))
	i += 1

print('* completed')
