import torch
import cv2
#TODO add automatic mask resolution changing
class Guide():
    
    def __init__(self,context, mask, mask_index, resolution, device, dtype):
        self.context = context
        self.mask = torch.tensor(cv2.resize(mask, (resolution, resolution)), dtype=dtype, device=device) / 255
        self.mask_index = mask_index
        self.resolution = resolution
        self.obj_attentions = []
        self.outputs = []
        self.modules = []
        self.step = 0
        self.device = device
        self.dtype = dtype
        self.indexes = []
    
    def reshape_heads_to_batch_dim(self, module, tensor):
            batch_size, seq_len, dim = tensor.shape
            head_size = module.heads
            tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
            tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
            return tensor

    def reshape_batch_dim_to_heads(self, module, tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = module.heads
                tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
                tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
                return tensor
    
    def fw_hook(self,module, input, output):
        self.outputs.append(output)
        #Add modules only if we are in the first step
        if self.step == 0:
            self.modules.append(module)


    def register_hook(self, net_, place_in_unet, module_name=None):
        if module_name in ["attn1", "attn2"]:
            net_.register_forward_hook(self.fw_hook)
        elif hasattr(net_, 'children'):
            for k, net__ in net_.named_children():
                self.register_hook(net__, place_in_unet, module_name=k)
        
    
    def guide(self):
        if self.step == 0:
            for l in range(len(self.outputs)):
                if self.outputs[l].shape[1] == self.resolution**2 \
                    and self.modules[l].to_v.in_features==768:
                        self.indexes.append(l)
            self.modules = [self.modules[i] for i in self.indexes]
        #Select correct modules and outputs
        self.outputs = [self.outputs[i] for i in self.indexes]
        
        #Generate target attention only for the first step because weights don't change
        with torch.no_grad():
            if self.step == 0:
                for attn_module in self.modules:
                    if len(self.context) == 2:
                        v = attn_module.to_v(self.context)
                    else:
                        v = attn_module.to_v(self.context)
                    
                    v = self.reshape_heads_to_batch_dim(attn_module, v)
                    
                    heads = attn_module.heads
                    if len(self.context==2):
                         heads = heads*2
                    obj_attn = torch.randn((attn_module.heads, self.resolution ** 2, 77), device=self.mask.device, dtype=self.mask.dtype) / 10
                    obj_attn[:, :, self.mask_index] = torch.stack([self.mask.reshape(-1)] * attn_module.heads)
                    
                    out = torch.einsum("b i j, b j d -> b i d", obj_attn, v)
                    
                    out = self.reshape_batch_dim_to_heads(attn_module, out)
                    out = attn_module.to_out[0](out)
                    self.obj_attentions.append(out)
                self.obj_attentions = torch.cat(self.obj_attentions).to(dtype=self.mask.dtype, device=self.mask.device)
        
        self.outputs = torch.cat(self.outputs).to(dtype=self.mask.dtype, device=self.mask.device)
        print(self.outputs.shape)

    def reset_step(self):
        self.outputs = []
        self.step += 1
        
