import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer
import time

class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_input_quant=False,
        rotate=True,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_buffer('weight',org_module.weight)
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape,rotate=rotate)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params,rotate=rotate)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False
        self.init_duquant_params = torch.tensor(0) if weight_quant_params['quant_method'] == 'duquant' else torch.tensor(1)

        self.stats = {
            "x_l2_diaq": [],
            "x_relnorm_diaq": [],
            "x_cos_diaq": [],
            "x_l2_rtn": [],
            "x_relnorm_rtn": [], 
            "x_cos_rtn": [],
            "wx_l2_diaq": [],
            "wx_relnorm_diaq": [],
            "wx_cos_diaq": [],
            "wx_l2_rtn": [],
            "wx_relnorm_rtn": [],
            "wx_cos_rtn": []
        }


    def forward(self, input: torch.Tensor, o_proj_yes=False):
        # print("input", input)
        if self.use_act_quant and not self.disable_input_quant:
            # rtn
            x_daq, x, x_rtn = self.act_quantizer(input)

            xlen=torch.norm(x, p=2, dim=-1, keepdim=True)

            l2_dist_rtn = torch.norm(x - x_rtn, p=2, dim=-1)
            rel_dist_rtn = l2_dist_rtn / (xlen.squeeze(-1) + 1e-8)
            cos_sim_rtn = F.cosine_similarity(x, x_rtn, dim=-1, eps=1e-8)

            self.stats["x_l2_rtn"].append(torch.mean(l2_dist_rtn.detach().cpu()).item())
            self.stats["x_relnorm_rtn"].append(torch.mean(rel_dist_rtn.detach().cpu()).item())
            self.stats["x_cos_rtn"].append(torch.mean(cos_sim_rtn.detach().cpu()).item())

            # diaq
            xqlen=torch.norm(x_daq, p=2, dim=-1, keepdim=True).unsqueeze(0)

            l2_dist_diaq = torch.norm(x - x_daq, p=2, dim=-1)
            rel_dist_diaq = l2_dist_diaq / (xlen.squeeze(-1) + 1e-8)
            cos_sim_diaq = F.cosine_similarity(x, x_daq, dim=-1, eps=1e-8)

            self.stats["x_l2_diaq"].append(torch.mean(l2_dist_diaq.detach().cpu()).item())
            self.stats["x_relnorm_diaq"].append(torch.mean(rel_dist_diaq.detach().cpu()).item())
            self.stats["x_cos_diaq"].append(torch.mean(cos_sim_diaq.detach().cpu()).item())

        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            if not self.init_duquant_params:
                self.weight_quantizer.copy_duquant_params(self.act_quantizer)
                self.init_duquant_params = torch.tensor(1) 
            weight = self.weight_quantizer(self.weight)[0]
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias
        
        input=x_daq
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        a = xqlen/xlen
        if not o_proj_yes:
            a = a.squeeze(0)
            out/=a
        
        # diaq
        wx = self.fwd_func(x, weight, bias, **self.fwd_kwargs)
        wx_q = out
        wx_rtn = self.fwd_func(x_rtn, weight, bias, **self.fwd_kwargs)

        wx_norm = torch.norm(wx, p=2, dim=-1)

        wx_l2_dist_diaq = torch.norm(wx - wx_q, p=2, dim=-1)
        wx_rel_dist_diaq = wx_l2_dist_diaq / (wx_norm + 1e-8)
        wx_cos_sim_diaq = F.cosine_similarity(wx, wx_q, dim=-1, eps=1e-8)

        wx_l2_dist_rtn = torch.norm(wx - wx_rtn, p=2, dim=-1)
        wx_rel_dist_rtn = wx_l2_dist_rtn / (wx_norm + 1e-8)
        wx_cos_sim_rtn = F.cosine_similarity(wx, wx_rtn, dim=-1, eps=1e-8)

        self.stats["wx_l2_diaq"].append(torch.mean(wx_l2_dist_diaq.detach().cpu()).item())
        self.stats["wx_relnorm_diaq"].append(torch.mean(wx_rel_dist_diaq.detach().cpu()).item())
        self.stats["wx_cos_diaq"].append(torch.mean(wx_cos_sim_diaq.detach().cpu()).item())

        self.stats["wx_l2_rtn"].append(torch.mean(wx_l2_dist_rtn.detach().cpu()).item())
        self.stats["wx_relnorm_rtn"].append(torch.mean(wx_rel_dist_rtn.detach().cpu()).item())
        self.stats["wx_cos_rtn"].append(torch.mean(wx_cos_sim_rtn.detach().cpu()).item())

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def copy_quantizers_duquant_params(self, proj):
        assert proj.init_duquant_params
        self.init_duquant_params = torch.tensor(1)
        self.weight_quantizer.copy_duquant_params(proj.weight_quantizer)
        self.act_quantizer.copy_duquant_params(proj.act_quantizer)


