
from config import *
from koni import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		self.vacant_board = nn.Parameter(self._get_vacant_board_(), requires_grad = False)
		self.embedding = nn.Parameter(self._get_embedding_())

		self.init_conv = Conv(input_channels, conv_channels, conv_kernel_size, True)
		
		self.tower = nn.ModuleList([
			Residual(conv_channels, conv_kernel_size) for _ in range(residual_blocks)
		])
		self.spi_head_conv = Conv(conv_channels, head_channels, 1, True)
		self.spi_hidden_fc = Dense(head_features, hidden_features, True)
		self.spi_fc = Dense(hidden_features, mini_board_size, False)

		self.epi_head_conv = Conv(conv_channels, head_channels, 1, True)
		self.epi_hidden_fc = Dense(head_features, hidden_features, True)
		self.epi_fc = Dense(hidden_features, mini_board_size, False)		

		self.v_head_conv = Conv(conv_channels, head_channels, 1, True)
		self.v_hidden_fc = Dense(head_features, hidden_features, True)
		self.v_fc = Dense(hidden_features, 1, False)

	def forward(self, x):
		len_x = x.shape[0]

		vacant_board = self.vacant_board.expand((len_x, -1, -1))
		embedding = self.embedding.expand((len_x, -1, -1))

		out = torch.cat((x, vacant_board, embedding), dim = 1)
		out = out.view(len_x, -1, mini_board_dim, mini_board_dim)

		out = self.init_conv(out)

		for _, block in enumerate(self.tower):
			out = block(out)

		v = self.v_head_conv(out)
		v = self.v_hidden_fc(v.flatten(1))
		v = self.v_fc(v)

		v = torch.tanh(v)
		v = v.squeeze(1)

		spi = self.spi_head_conv(out)
		spi = self.spi_hidden_fc(spi.flatten(1))
		spi = self.spi_fc(spi)
		spi = F.log_softmax(spi, dim = -1)

		epi = self.epi_head_conv(out)
		epi = self.epi_hidden_fc(epi.flatten(1))
		epi = self.epi_fc(epi)
		epi = F.log_softmax(epi, dim = -1)

		spi = spi.reshape((len_x, mini_board_size, 1)).expand((-1, -1, mini_board_size))
		epi = epi.reshape((len_x, 1, mini_board_size)).expand((-1, mini_board_size, -1))

		pi = spi + epi
		return pi, v

	def _get_vacant_board_(self):
		np_arr = vacant_mini_board[np.newaxis, np.newaxis, :] + 2
		return torch.FloatTensor(np_arr)

	def _get_embedding_(self):
		np_arr = value_mini_board[np.newaxis, :, :] / quad_dim
		return torch.FloatTensor(np_arr)

class Residual(nn.Module):
	def __init__(self, channels, kernel_size):
		super(Residual, self).__init__()

		self.conv1 = Conv(channels, channels, kernel_size, True)
		self.conv2 = Conv(channels, channels, kernel_size, False)

	def forward(self, x):
		out = self.conv1(x)
		out = self.conv2(out)
		out = F.relu(out + x)
		return out

class Conv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, apply_relu):
		super(Conv, self).__init__()

		self.layer = nn.Conv2d(
			in_channels,
			out_channels,
			kernel_size,
			1,
			(kernel_size - 1) // 2
		)
		self.bn = nn.BatchNorm2d(out_channels)
		self.apply_relu = apply_relu

	def forward(self, x):
		out = self.layer(x)
		out = self.bn(out)
		if self.apply_relu: out = F.relu(out)
		return out

class Dense(nn.Module):
	def __init__(self, in_features, out_features, apply_relu):
		super(Dense, self).__init__()

		self.layer = nn.Linear(in_features, out_features)
		self.apply_relu = apply_relu

	def forward(self, x):
		out = self.layer(x)
		if self.apply_relu: out = F.relu(out)
		return out
