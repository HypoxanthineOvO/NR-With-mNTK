import torch
from NTK import GetFuncParams, Evaluate_NTK

Module_Eval_NTK = {
    "Linear": "linear",
    "Conv2d": "conv",
}

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()


        self.module_lists = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=7*7*32, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=10)
        ])

        self.num_modules = len(self.module_lists)

    def forward(self, x: torch.Tensor):
        for i, module in enumerate(self.module_lists):
            x = module(x)
        return x
        
    def forward_with_evaluate_ntk(self, x: torch.Tensor):
        ntks = {}
        module_count = {}  # 记录每种模块出现的次数
        
        for i, module in enumerate(self.module_lists):
            module_name = module.__class__.__name__
            
            # 为模块生成唯一名称（如 Conv2d_0, Conv2d_1）
            if module_name not in module_count:
                module_count[module_name] = 0
            else:
                module_count[module_name] += 1
            unique_name = f"{module_name}_{module_count[module_name]}"
            
            # 计算NTK（仅限支持的模块类型）
            if module_name in Module_Eval_NTK:
                func, params = GetFuncParams(module, model_type=Module_Eval_NTK[module_name])
                ntk = Evaluate_NTK(func, params, x, x, compute='mNTK')
                ntks[unique_name] = ntk  # 使用唯一名称作为键
            
            x = module(x)
        
        return x, ntks

    def get_module_eval_ntk_name(self):
        module_names = []
        module_count = {}  # 记录每种模块出现的次数
        
        for i, module in enumerate(self.module_lists):
            module_name = module.__class__.__name__
            
            # 为模块生成唯一名称
            if module_name not in module_count:
                module_count[module_name] = 0
            else:
                module_count[module_name] += 1
            unique_name = f"{module_name}_{module_count[module_name]}"
            
            if module_name in Module_Eval_NTK:
                module_names.append(unique_name)
        
        return module_names

