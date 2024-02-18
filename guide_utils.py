import torch
import cv2
#TODO add automatic mask resolution changing
class Guide():
    
    def __init__(self,context, mask, mask_index, resolution, device, dtype, guidance_scale, base_masks=None, diffusion_type="LCM"):
        
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
        self.guidance_scale = guidance_scale
        self.base_masks = base_masks
        self.diffusion_type = diffusion_type
    
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


    def register_hook(self, net_, count, place_in_unet, module_name=None):
        if module_name in ["attn1", "attn2"]:
            net_.register_forward_hook(self.fw_hook)
            return count + 1
        elif hasattr(net_, 'children'):
            for k, net__ in net_.named_children():
             count = self.register_hook(net__, count, place_in_unet, module_name=k)
        self.count = count
        return count
    
    def guide(self):
        if self.step == 0:
            
            for l in range(self.count):
                if self.outputs[l].shape[1] == self.resolution**2 \
                    and self.modules[l].to_v.in_features==768:
                        self.indexes.append(l)
            self.modules = [self.modules[i] for i in self.indexes]
            self.count = len(self.modules)
        
        out_indexes = self.indexes.copy()
        if self.diffusion_type == "SD":
             
             out_indexes.extend([(i + (len(self.outputs)//2)) for i in self.indexes])
             
        #Select correct modules and outputs
        
        self.outputs = [self.outputs[i] for i in out_indexes]
        
        
        #Generate target attention only for the first step because weights don't change
        with torch.no_grad():
            if self.step == 0:
                o_unconds = []
                o_texts = []
                for attn_module in self.modules:
                    if len(self.context) == 2:
                        v_uncond = attn_module.to_v(self.context[0].unsqueeze(0))
                        v_uncond = self.reshape_heads_to_batch_dim(attn_module, v_uncond)
                        v_text = attn_module.to_v(self.context[1].unsqueeze(0))
                        v_text = self.reshape_heads_to_batch_dim(attn_module, v_text)
                        
                        obj_attn = torch.randn((attn_module.heads, self.resolution ** 2, 77), device=self.mask.device, dtype=self.mask.dtype) / 10
                        obj_attn[:, :, self.mask_index] = torch.stack([self.mask.reshape(-1)] * attn_module.heads)
                        
                        out = torch.einsum("b i j, b j d -> b i d", obj_attn, v_uncond)
                        out = self.reshape_batch_dim_to_heads(attn_module, out)
                        out_uncond = attn_module.to_out[0](out)

                        out = torch.einsum("b i j, b j d -> b i d", obj_attn, v_text)
                        out = self.reshape_batch_dim_to_heads(attn_module, out)
                        out_text = attn_module.to_out[0](out)

                        o_unconds.append(out_uncond)
                        o_texts.append(out_text)
                    else:
                        v = attn_module.to_v(self.context)
                        v = self.reshape_heads_to_batch_dim(attn_module, v)
                        
                        obj_attn = torch.randn((attn_module.heads, self.resolution ** 2, 77), device=self.mask.device, dtype=self.mask.dtype) / 50
                        if self.base_masks != None:
                            for m in range(len(self.base_masks)):
                                obj_attn[:, :, m] = torch.stack([self.base_masks[m].reshape(-1)] * attn_module.heads) / 10
                        
                        obj_attn[:, :, self.mask_index] = torch.stack([self.mask.reshape(-1)] * attn_module.heads)
                        
                        out = torch.einsum("b i j, b j d -> b i d", obj_attn, v)
                        
                        out = self.reshape_batch_dim_to_heads(attn_module, out)
                        out = attn_module.to_out[0](out)
                        self.obj_attentions.append(out)
                if len(self.context) == 2:
                    self.obj_attentions = [torch.cat(o_unconds).to(dtype=self.mask.dtype, device=self.mask.device), 
                                            torch.cat(o_texts).to(dtype=self.mask.dtype, device=self.mask.device)]
                    
                else:
                    self.obj_attentions = torch.cat(self.obj_attentions).to(dtype=self.mask.dtype, device=self.mask.device)
        
        self.outputs = torch.cat(self.outputs).to(dtype=self.mask.dtype, device=self.mask.device)
        

    def reset_step(self):
        del self.outputs
        self.outputs = []
        self.step += 1
        
