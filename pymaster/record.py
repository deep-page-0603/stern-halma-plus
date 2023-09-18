
class Record(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.total = 0
		self.wins = 0
		self.loses = 0
		self.draws = 0
		self.arrests = 0
		self.prisons = 0
		self.depth_outs = 0
		self.depth = 0

	def __str__(self):
		s =  '\t\t# of total : {}\n'.format(self.total)
		s += '\t\t# of wins : {}\n'.format(self.wins)
		s += '\t\t# of loses : {}\n'.format(self.loses)
		s += '\t\t# of draws : {}\n'.format(self.draws)
		s += '\t\t# of arrests : {}\n'.format(self.arrests)
		s += '\t\t# of prisons : {}\n'.format(self.prisons)
		s += '\t\t# of depth_outs : {}\n'.format(self.depth_outs)

		if self.total > 0: s += '\t\tavg depth : {}\n'.format(self.depth // self.total)
		return s

	def add(self, reward, exit, depth):
		self.total += 1
		self.depth += depth

		if reward > 0:
			self.wins += 1
		elif reward < 0:
			self.loses += 1
		else:
			self.draws += 1

		if exit == 1:
			if reward >= 0: self.arrests += 1
			if reward <= 0: self.prisons += 1
		elif exit == 2:
			self.depth_outs += 1

	def get_margin(self):
		if self.total == 0: return 0
		return (self.wins - self.loses) / self.total
