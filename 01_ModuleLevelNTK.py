import torch
from NTK import Evaluate_NTK, GetFuncParams

if __name__ == "__main__":
    # 1. Linear Model's NTK
    linear_model = torch.nn.Linear(10, 5)
    x_linear_input = torch.randn(3, 10)
    linear_func, linear_params = GetFuncParams(linear_model, model_type="linear")
    linear_ntk = Evaluate_NTK(linear_func, linear_params, x_linear_input, x_linear_input, compute='mNTK')
    linear_result = linear_model(x_linear_input)
    print("=" * 50)
    print(f"Linear Model NTK Shape: {linear_ntk.shape}")
    print(f"Linear Model Output Shape: {linear_result.shape}")

    # 2. Convolutional Model's NTK
    NUM_INPUT_CHANNELS = 3
    NUM_OUTPUT_CHANNELS = 1
    conv_model = torch.nn.Conv2d(in_channels=NUM_INPUT_CHANNELS, out_channels=NUM_OUTPUT_CHANNELS, kernel_size=3, stride=1, padding=1)
    x_conv_input = torch.randn(3, NUM_INPUT_CHANNELS, 5, 5)
    conv_func, conv_params = GetFuncParams(conv_model, model_type="conv")
    conv_ntk = Evaluate_NTK(conv_func, conv_params, x_conv_input, x_conv_input, compute='mNTK')
    conv_result = conv_model(x_conv_input)
    print("=" * 50)
    print(f"Convolutional Model NTK Shape: {conv_ntk.shape}")
    print(f"Convolutional Model Output Shape: {conv_result.shape}")

    # 3. Attention Model's NTK
    # attention_model: torch.nn.MultiheadAttention = torch.nn.MultiheadAttention(embed_dim=10, num_heads=2)
    # x_attention_input = torch.randn(3, 5, 10)  # (sequence_length, batch_size, embedding_dim)
    # attention_func, attention_params = GetFuncParams(attention_model, model_type="attention")
    # attention_ntk = Evaluate_NTK(attention_func, attention_params, x_attention_input, x_attention_input, compute='mNTK')
    # attention_result = attention_model(
    #     query = x_attention_input, key = x_attention_input, value = x_attention_input
    # )[0]
    # print("=" * 50)
    # print(f"Attention Model NTK Shape: {attention_ntk.shape}")
    # print(f"Attention Model Output Shape: {attention_result.shape}")