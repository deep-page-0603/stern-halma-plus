
# koni configuration
dim = 5
dim_1 = dim - 1
dim_2 = dim - 2
dbl_dim = 2 * dim_1
tri_dim = 3 * dim_1
quad_dim = 4 * dim_1
pentad_dim = 5 * dim_1
board_dim = quad_dim + 1
board_size = board_dim * board_dim
mini_board_dim = dbl_dim + 3
mini_board_size = mini_board_dim * mini_board_dim
piece_count = sum([n + 1 for n in range(dim)])
max_game_depth = 200

# model configuration
input_channels = 6
conv_kernel_size = 3
conv_channels = 80
residual_blocks = mini_board_dim - 1
head_channels = 4
head_features = head_channels * mini_board_size
hidden_features = 2 * mini_board_size

# selfplay configuration
noise_factor = 0.25
dirichlet_alpha = 0.03
warm_depth = 20
cold_temper = 1e-3

# tree-search configuration
num_playouts = 1600
p_uct = 3.0

# train configuration
batch_size = 2048
init_lr = 1e-4
l2_const = 1e-4
samples_per_train = 2048
valid_freq = 200
valid_games = 40
margin_thres = 0.8
