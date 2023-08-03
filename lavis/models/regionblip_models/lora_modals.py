import math
import loralib as lora 
import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(lora.Linear):

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        lora.LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out

        # Actual trainable parameters
        if r > 0:
            self.lora_A_img = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B_img = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.lora_A_img_region = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B_img_region = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.lora_A_pc = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B_pc = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.lora_A_pc_region = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B_pc_region = nn.Parameter(self.weight.new_zeros((out_features, r)))

            self.scaling = self.lora_alpha / self.r

            # TODO: Freezing the pre-trained weight matrix
            #self.weight.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A_img'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A_img, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_img)
        
        if hasattr(self, 'lora_A_img_region'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A_img_region, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_img_region)
        
        if hasattr(self, 'lora_A_pc'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A_pc, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_pc)
        
        if hasattr(self, 'lora_A_pc_region'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A_pc_region, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_pc_region)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        assert not self.merged
        if mode:
            #if self.merge_weights and self.merged:
            #    # Make sure that the weights are not merged
            #    if self.r > 0:
            #        self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
            #    self.merged = False
            pass
        else:
            #if self.merge_weights and not self.merged:
            #    # Merge the weights and mark it
            #    if self.r > 0:
            #        self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
            #    self.merged = True       
            pass

    def forward(self, x, modal_name):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        #if self.r > 0 and not self.merged:
        #    result = F.linear(x, T(self.weight), bias=self.bias)            
        #    result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        #    return result
        #else:
        #    return F.linear(x, T(self.weight), bias=self.bias)

        result = F.linear(x, T(self.weight), bias=self.bias) 
        if self.r > 0:
            if modal_name == 'image':
                result += (self.lora_dropout(x) @ self.lora_A_img.transpose(0, 1) @ self.lora_B_img.transpose(0, 1)) * self.scaling
            elif modal_name == 'image_region':
                result += (self.lora_dropout(x) @ self.lora_A_img_region.transpose(0, 1) @ self.lora_B_img_region.transpose(0, 1)) * self.scaling
            elif modal_name == 'pointcloud':
                result += (self.lora_dropout(x) @ self.lora_A_pc.transpose(0, 1) @ self.lora_B_pc.transpose(0, 1)) * self.scaling
            elif modal_name == 'pointcloud_region':
                result += (self.lora_dropout(x) @ self.lora_A_pc_region.transpose(0, 1) @ self.lora_B_pc_region.transpose(0, 1)) * self.scaling
            else:
                raise ValueError('Not supported modal_name: {}'.format(modal_name))
        return result
    