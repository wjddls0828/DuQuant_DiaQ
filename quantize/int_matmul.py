import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer


class QuantMatMul(nn.Module):
    def __init__(
        self,
        x1_quant_params: dict = {},
        x2_quant_params: dict = {},
        disable_act_quant=False,
        matmul_func=torch.bmm,
        rotate=True,
    ):
        super().__init__()
        # de-activate the quantized forward default
        self.use_act_quant = False
        # initialize quantizer
        self.i_cluster_counts = None
        self.x1_quantizer = UniformAffineQuantizer(**x1_quant_params,rotate=rotate)
        self.x2_quantizer = UniformAffineQuantizer(**x2_quant_params,rotate=rotate)
        self.matmul_func = matmul_func

        self.disable_act_quant = disable_act_quant


    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def quant_x1(self, x1, attn=False):
        if self.use_act_quant:
            x1 = self.x1_quantizer(x1)
            if not attn:
                x1, x1_orig, _ =x1
                return x1, x1_orig
            else:
                return x1

    def quant_x2(self, x2):
        if self.use_act_quant:
            x2, x2_orig, _ = self.x2_quantizer(x2)
        return x2, x2_orig

    def forward(self, x1, x2):
        out = self.matmul_func(x1, x2)
        return out
