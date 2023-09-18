
from tqdm import tqdm
from config import *
from koni import *
from util import *
from net import *
from players import *
from compete import *
import numpy as np
import time
import glob
import argparse
import random
import os

def load_file_data(data_dir):
	files = glob.glob(data_dir + '/*.dat')
	data = []

	for f in tqdm(files, desc = 'loading', leave = False):
		pd = pickle_load(f)

		for i, d in enumerate(pd):
			data.append((i % 2, d))

	return data

def get_train_samples(data):
	samples = []

	for t, d in data:
		s, pi_info, v = d
		full_pi = np.zeros((mini_board_size, mini_board_size), dtype = np.float32)

		for spt, ept, p in pi_info:
			full_pi[spt, ept] = p

		if len(s) == 2:
			lc = np.zeros((1, mini_board_size), dtype = np.int8)
			if t == 0: lc[:, :] = 1
			s = np.vstack([s, lc])			

		samples.append((s, full_pi, v))

	return samples

def main(args):
	stage = args.stage

	log_dir = './logs/stage_{}'.format(stage)
	data_dir = './data/stage_{}'.format(stage)
	log_file = log_dir + '/__log__.txt'
	
	def _log_(msg = ''):
		bilog(log_file, msg)

	bkup_dir = log_dir + '/src_bkup'	
	backup_src(bkup_dir)

	last_log_file = './logs/best.ckpt'
	epoch = 0
	max_margin = 0

	if not args.new:
		log_files = glob.glob(log_dir + '/*.ckpt')

		for f in sorted(log_files):
			if f.endswith('best.ckpt'): continue

			last_log_file = f.replace('\\', '/')
			segs = last_log_file.split('/')[-1][:-5].split('_')

			epoch = int(segs[0].split('=')[1])
			margin = float(segs[1].split('=')[1])

			if margin > max_margin: max_margin = margin
	else:
		rmdir(log_dir)

	agent_net = Net(last_log_file)
	env_net = Net('./logs/best.ckpt')	
	
	agent_player = MCTSPlayer(net = agent_net, playouts = num_playouts, p_factor = p_uct, mode = 'valid')
	env_player = MCTSPlayer(net = env_net, playouts = num_playouts, p_factor = p_uct, mode = 'valid')

	_log_()
	_log_('* train {} on {}'.format(
		'started' if epoch == 0 else 'resumed',
		time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
	))
	_log_()

	if args.valid:
		record = compete([agent_player, env_player], valid_games)[0]
		print()
		print(str(record))
		print()

	file_data = load_file_data(data_dir)

	while True:
		for _ in tqdm(range(valid_freq), desc = 'training', leave = False):
			train_samples = get_train_samples(random.sample(file_data, samples_per_train))
			epoch += 1

			pi_loss, v_loss = agent_net.train(train_samples)
			loss = pi_loss + v_loss
		
		record = compete([agent_player, env_player], valid_games)[0]
		margin = record.get_margin()

		agent_net.save('{}/epoch={:06}_margin={:.2f}_loss={:.4f}.ckpt'.format(
			log_dir, epoch, margin, loss
		))
		_log_('epoch-{:06}: margin={:.2f}, pi_loss={:.4f}, v_loss={:.4f}, loss = {:.4f}'.format(
			epoch, margin, pi_loss, v_loss, loss
		))

		if margin > max_margin:
			agent_net.save('{}/best.ckpt'.format(log_dir))
			max_margin = margin

		_log_()
		_log_(str(record))
		_log_()

		if margin > margin_thres and args.stage > 0:
			_log_('* stopped for margin reached : {:.2f}'.format(margin))
			break

if __name__ == '__main__':	
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--stage', type = int, help = 'stage')
	parser.add_argument('-n', '--new', action = 'store_true', default = False, help = 'new start')
	parser.add_argument('-v', '--valid', action = 'store_true', default = False, help = 'validation first')
	args = parser.parse_args()

	if args.stage is None:
		print('# invalid stage')
		exit()

	main(args)
