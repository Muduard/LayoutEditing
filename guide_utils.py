import torch
import cv2
from queue import Queue
from ptp_utils import save_tensor_as_image

class Guide():
    
    def __init__(self,context, masks, mask_indexes, resolution, device, dtype, guidance_scale, base_masks=None, diffusion_type="LCM", init_type="a"):
        
        self.context = context
        acceptable_masks_indexes = []
        for i in range(len(mask_indexes)):
             if mask_indexes[i] < 77:
                  acceptable_masks_indexes.append(i)
        
        self.mask_indexes = [mask_indexes[i] for i in acceptable_masks_indexes]
        self.masks = []
        for i in acceptable_masks_indexes:
             self.masks.append(torch.tensor(cv2.resize(masks[i], (resolution, resolution)), 
                                            dtype=dtype, device=device) / 255)
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
        self.init_type = init_type
    
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
        


    def register_hook(self, model, count, place_in_unet, module_name=None):
        queue = Queue()
        queue.put( (module_name, model) )

        self.hooks = []

        while not queue.empty():
            module_name, module = queue.get()
            
            if module_name in ["attn1", "attn2"]:
                self.hooks.append(module.register_forward_hook(self.fw_hook))  
            elif hasattr(module, 'children'):
                for module_name, child_module in module.named_children():
                    queue.put( (module_name, child_module) )
            
        
        self.count = len(self.hooks)
        
        
    
    def guide(self):
        if self.step == 0:
            
            for l in range(self.count):
                if self.outputs[l].shape[1] == self.resolution**2 \
                    and self.modules[l].to_v.in_features==768:
                        self.indexes.append(l)
            self.modules = [self.modules[i] for i in self.indexes]
            self.count = len(self.modules)
        
        out_indexes = self.indexes.copy()
        #if self.diffusion_type == "SD":
             
             #out_indexes.extend([(i + (len(self.outputs)//2)) for i in self.indexes])
             
        #Select correct modules and outputs
        self.outputs = [self.outputs[i] for i in out_indexes]
        
        #Generate target attention only for the first step because weights don't change
        with torch.no_grad():
            if self.step == 0:
                
                for i, attn_module in enumerate(self.modules):
                    if self.init_type == "copy":
                        v = attn_module.to_v(self.context).to(torch.float)
                        vt = v.permute(0,2,1)
                        o_i = self.outputs[i][0]
                        #o_i = o_i - attn_module.to_out[0].bias
                        w = attn_module.to_out[0].weight.unsqueeze(0)
                        o_i = o_i.T.unsqueeze(0).to(torch.float)
                        w = w.to(torch.float)

                        ob_i = torch.linalg.lstsq(w, o_i).solution.to(torch.float)
                        attn = torch.linalg.lstsq(vt, ob_i).solution
                        for j, mask_index in enumerate(self.mask_indexes):
                            attn[:, mask_index, :] = self.masks[j].reshape(-1)
                        print(self.mask_indexes)
                        for j in range(8):
                            a = attn[0,j,:].reshape((32,32))
                            save_tensor_as_image(a, f"{j}.png")
                        attn = attn.permute(0,2,1)
                        
                        
                        out = attn @ v
                        
                        out = out.to(torch.float16)
                        
                    else:
                        v = attn_module.to_v(self.context)
                        v = self.reshape_heads_to_batch_dim(attn_module, v)
                        heads = attn_module.heads * len(self.context)
                        obj_attn = torch.zeros((heads, self.resolution ** 2, 77), device=self.device, dtype=self.dtype)
                        
                        for i, mask_index in enumerate(self.mask_indexes):
                            obj_attn[:, :, mask_index] = torch.stack([self.masks[i].reshape(-1)] * heads)
                        
                        out = torch.einsum("b i j, b j d -> b i d", obj_attn, v)
                        out = self.reshape_batch_dim_to_heads(attn_module, out)

                    out = attn_module.to_out[0](out)
                        
                    self.obj_attentions.append(out)

                self.obj_attentions = torch.cat(self.obj_attentions).to(dtype=self.dtype, device=self.device)
        
        self.outputs = torch.cat(self.outputs).to(dtype=self.dtype, device=self.device)
        

    def reset_step(self):
        del self.outputs
        self.outputs = []
        self.step += 1

    def reset(self):
        del self.outputs
        for hook in self.hooks:
             hook.remove()