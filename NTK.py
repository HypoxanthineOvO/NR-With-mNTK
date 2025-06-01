import torch

def GetFuncParams(model: torch.nn.Module, model_type: str = "linear"):
    params = dict(model.named_parameters())
    def model_func_linear(params, x: torch.Tensor):
        # Linear Model: [NI] -> [NO] for single input
        assert x.dim() == 1, "Input tensor must be a single 1D tensor for linear model."
        return torch.func.functional_call(model, params, (x.unsqueeze(0), )).squeeze(0)
    def model_func_conv(params, x: torch.Tensor):
        # Convolutional Model: [C, H, W] -> [NO] for single input
        assert x.dim() == 3, "Input tensor must be a single 3D tensor for convolutional model."
        return torch.func.functional_call(model, params, (x, )).flatten()
    def model_func_attention(params, x: torch.Tensor):
        # print(f"Input shape for attention model: {x.shape}")
        return torch.func.functional_call(model, params, (x )).squeeze(0)

    if model_type == "linear":
        model_func = model_func_linear
    elif model_type == "conv":
        model_func = model_func_conv
    elif model_type == "attention":
        raise NotImplementedError("Attention model is not implemented yet.")
        model_func = model_func_attention
    else:
        raise ValueError(f"Unknown model type: {model_type}. Expected 'linear', 'conv', or 'attention'.")
    return model_func, params

def Evaluate_NTK(
        func: callable, params: dict, 
        x1: torch.Tensor, x2: torch.Tensor, 
        compute: str = 'full'
    ) -> torch.Tensor:
    # X1 and X2 Shape:
    # print(f"x1 shape: {x1.shape}, x2 shape: {x2.shape}")
    def get_ntk(x1, x2):
        def func_x1(params):
            return func(params, x1)
        def func_x2(params):
            return func(params, x2)
        output: torch.Tensor
        vjp_fn: callable
        output, vjp_fn = torch.func.vjp(func_x1, params)
        def get_ntk_slice(vec):
            vjps = vjp_fn(vec)
            _, jvps = torch.func.jvp(func_x2, (params,), vjps)
            return jvps

        basis = torch.eye(output.numel(), dtype=output.dtype, device=output.device).view(output.numel(), -1)
        return torch.func.vmap(get_ntk_slice)(basis)
    
    result = torch.func.vmap(torch.func.vmap(get_ntk, (None, 0)), (0, None))(x1, x2) 
    # print(result.shape)
    if compute == "mNTK":
        N, M, K, _ = result.shape
        # 转换为 (N*K, M*K) 的形状
        result = result.permute(0, 2, 1, 3).contiguous()
        result = result.view(N * K, M * K)
        return result
    elif compute == 'full':
        return result
    elif compute == 'trace':
        return torch.einsum('NMKK->NM', result)
    elif compute == 'diagonal':
        return torch.einsum('NMKK->NMK', result)
    else:
        raise ValueError(f"Unknown compute type: {compute}. Expected 'full', 'trace', or 'diagonal'.")
