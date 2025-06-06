import torch
import json
from NTK import Evaluate_NTK, Evaluate_NTK_Multiple, GetFuncParams
from Modules import Hash
from NGP import InstantNGP
from BERT import MultiHeadAttention

if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# 1. Linear Model's NTK
	# linear_model = torch.nn.Linear(10, 5)
	# x_linear_input = torch.randn(3, 10)
	# linear_func, linear_params = GetFuncParams(linear_model, model_type="linear")
	# linear_ntk = Evaluate_NTK(linear_func, linear_params, x_linear_input, x_linear_input, compute='mNTK')
	# linear_result = linear_model(x_linear_input)
	# print("=" * 50)
	# print(f"Linear Model NTK Shape: {linear_ntk.shape}")
	# print(f"Linear Model Output Shape: {linear_result.shape}")

	# 2. Convolutional Model's NTK
	# NUM_INPUT_CHANNELS = 3
	# NUM_OUTPUT_CHANNELS = 1
	# conv_model = torch.nn.Conv2d(in_channels=NUM_INPUT_CHANNELS, out_channels=NUM_OUTPUT_CHANNELS, kernel_size=3, stride=1, padding=1)
	# x_conv_input = torch.randn(3, NUM_INPUT_CHANNELS, 5, 5)
	# conv_func, conv_params = GetFuncParams(conv_model, model_type="conv")
	# conv_ntk = Evaluate_NTK(conv_func, conv_params, x_conv_input, x_conv_input, compute='mNTK')
	# conv_result = conv_model(x_conv_input)
	# print("=" * 50)
	# print(f"Convolutional Model NTK Shape: {conv_ntk.shape}")
	# print(f"Convolutional Model Output Shape: {conv_result.shape}")

	# 3. Attention Model's NTK
	attention_model = torch.nn.MultiheadAttention(30, 2)
	x_attention_input = torch.randn(1, 30, 30)  # (sequence_length, batch_size, embedding_dim)

	attention_mask = torch.ones(30, 1, dtype = torch.bool)  # (sequence_length, batch_size, sequence_length)
	
	attention_params = dict(attention_model.named_parameters())
	def model_function_attention(params, x: torch.Tensor):
		q,k,v = x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0)
		return torch.func.functional_call(attention_model, params, (q,k,v, attention_mask))

	attention_ntk = Evaluate_NTK(
		model_function_attention, attention_params, 
		x_attention_input,
		x_attention_input, 
		compute='mNTK')
	print("=" * 50)
	print(f"Attention Model NTK Shape: {attention_ntk.shape}")

	# 4. Hash Encoding Module
#   hash_config = 	{
# 	"otype": "HashGrid",
# 	"n_levels": 16,
# 	"n_features_per_level": 2,
# 	"log2_hashmap_size": 19,
# 	"base_resolution": 16,
# 	"per_level_scale": 1.38191288
# }
#   hash_model = Hash.HashEncoding(3, hash_config)
#   x_hash_input = torch.randn(2, 3)  # (batch_size, input_dim)
#   func_hash, params_hash = GetFuncParams(hash_model, model_type="hash")
#   #print(func_hash, params_hash, params_hash.keys())
#   hash_ntk = Evaluate_NTK(func_hash, params_hash, x_hash_input, x_hash_input, compute='mNTK')
#   hash_result = hash_model(x_hash_input)
#   print("=" * 50)
#   print(f"Hash Encoding Model NTK Shape: {hash_ntk.shape}")
#   print(f"Hash Encoding Model Output Shape: {hash_result.shape}")

	# 5. Instant NGP
	# with open("./configs/base.json", "r") as f:
	# 		config = json.load(f)
	# ngp = InstantNGP(config, True).to(device)
	# ngp_input_position = torch.randn(2, 3).to(device)  # (batch_size, input_dim)
	# ngp_input_direction = torch.randn(2, 3).to(device)  # (batch_size, input_dim)
	# _, ntk_records = ngp.forward_with_evaluate_ntk(ngp_input_position, ngp_input_direction)
	# print("=" * 50)
	# print(ntk_records)