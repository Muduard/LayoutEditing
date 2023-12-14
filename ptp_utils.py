import abc
from typing import Optional, Union, Tuple, List, Callable, Dict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
LOW_RESOURCE = True 
class LocalBlend:

    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02, file_path="Attns.png"):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    plt.imshow(pil_img)
    plt.show()
    plt.savefig(file_path)

class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):

    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        #if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
        self.step_store[key].append(attn)

        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionReplace(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionReplace, self).forward(attn, is_cross, place_in_unet)
        if attn.shape[2] == 77:
          if f'{attn.shape[1]}' in self.new_attn.keys():
            attn[:, :, self.token_numbers] = self.new_attn[f'{attn.shape[1]}']
        return attn

    def __init__(self, new_attn, token_numbers):
        super(AttentionReplace, self).__init__()
        self.new_attn = new_attn
        self.token_numbers = token_numbers

def aggregate_attention(prompts, attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2

    for location in from_where:

        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out_save = out.copy()
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out, out_save

@torch.no_grad()
def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    image = (image * 255).astype(np.uint8)
    return image



def get_multi_level_attention_from_average(average_attention, device):
    # the image to torch tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    h8 = cv2.resize(average_attention, (8, 8))
    h16 = cv2.resize(average_attention, (16, 16))
    h32 = cv2.resize(average_attention, (32, 32))
    h64 = cv2.resize(average_attention, (64, 64))
    # mask =
    '''noise_h = np.random.normal(0, 100, average_attention.shape).astype(np.uint8)
    noise_h32 = np.random.normal(0, 100, h32.shape).astype(np.uint8)
    noise_h64 = np.random.normal(0, 100, h64.shape).astype(np.uint8)
    image = cv2.bitwise_and(noise_h, average_attention)
    image32 = cv2.bitwise_and(noise_h32, h32)
    image64 = cv2.bitwise_and(noise_h64, h64)'''
    image8 = h8
    image16 = h16
    image32 = h32
    image64 = h64

    image_tensor8 = transform(image8).to(dtype=torch.float16, device=device) / 255
    image_tensor8 = image_tensor8.repeat((8, 1, 1)).reshape(8, 64, 1)

    image_tensor16 = transform(image16).to(dtype=torch.float16, device=device) / 255
    image_tensor16 = image_tensor16.repeat((8, 1, 1)).reshape(8, 256, 1)

    image_tensor32 = transform(image32).to(dtype=torch.float16, device=device) / 255
    image_tensor32 = image_tensor32.repeat((8, 1, 1)).reshape(8, 1024, 1)

    image_tensor64 = transform(image64).to(dtype=torch.float16, device=device) / 255
    image_tensor64 = image_tensor64.repeat((8, 1, 1)).reshape(8, 4096, 1)

    attn_new = {'64': image_tensor8,'256': image_tensor16, '1024': image_tensor32, '4096': image_tensor64}

    return attn_new

def find_layers_by_name(model, target_names):
    # Initialize an empty list to store layers with the specified name
    found_layers = {"": [],"down": [], "mid": [], "up": []}

    # Define a function to recursively search for layers with the target name
    def find_layers_recursive(module, name, place):
        for child_name, child_module in module.named_children():
            if name in child_name:
                found_layers[place].append(child_module)
            if place != "":
                find_layers_recursive(child_module, name, place)
            else:
                if child_name == "down_blocks":
                    find_layers_recursive(child_module, name, "down")
                elif child_name == "mid_block":
                    find_layers_recursive(child_module, name, "mid")
                elif child_name == "up_blocks":
                    find_layers_recursive(child_module, name, "up")
                else:
                    find_layers_recursive(child_module, name, "")

    # Start the search from the top-level model
    for target_name in target_names:
      find_layers_recursive(model, target_name, "")

    return found_layers

def get_cross_attention(prompts, attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    attention_maps, out_save = aggregate_attention(prompts,attention_store, res, from_where, True, select)

    return attention_maps, out_save

def show_cross_attention(tokenizer, prompts, attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0, out_path: str = "attns.png"):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps, out_save = aggregate_attention(prompts,attention_store, res, from_where, True, select)

    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.cpu().detach().numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        impil = Image.fromarray(image).resize((256, 256))
        impil.save(f"attentions/{decoder(int(tokens[i]))}.png")
        image = text_under_image(image, decoder(int(tokens[i])))

        images.append(image)
    view_images(np.stack(images, axis=0),file_path=out_path)

kl = torch.nn.KLDivLoss()
def diffusion_step(unet, scheduler, controller, latents, context, t, guidance_scale, xt = None, m=None, train = False,guide=False, sigma=1):
    
    latents_input = torch.cat([latents] * 2)
    latents_input = scheduler.scale_model_input(latents_input, t)
    noise_pred = unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)

    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

    latents = scheduler.step(noise_pred, t, latents)["prev_sample"]
    
    if controller != None:
        latents = controller.step_callback(latents)
        
    
    return latents, noise_pred

def register_attention_control(model, controller, attns = None):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        
        #self.emb_model = EmbedderModel().to(dtype=torch.float16, device="cuda:1")
        def reshape_heads_to_batch_dim(self, tensor):
            batch_size, seq_len, dim = tensor.shape
            head_size = self.heads
            tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
            tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
            return tensor

        def reshape_batch_dim_to_heads(self, tensor):
            batch_size, seq_len, dim = tensor.shape
            head_size = self.heads
            tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
            tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
            return tensor

        # def forward(x, context=None, mask=None):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            x = hidden_states
            context = encoder_hidden_states
            mask = attention_mask

            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            
            k = self.to_k(context)
            v = self.to_v(context)
            q = reshape_heads_to_batch_dim(self, q)
            k = reshape_heads_to_batch_dim(self, k)
            v = reshape_heads_to_batch_dim(self, v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            additional_attn = 0
            w = 0.5 * sim.max()
            #pww
            if attns != None:
                additional_attn = attns[f'{sim.shape[1]}']

            # attention, what we cannot get enough of
            attn = F.softmax(sim + w * additional_attn,dim=-1)
            
            attn = controller(attn, is_cross, place_in_unet)

            out = torch.einsum("b i j, b j d -> b i d", attn, v)

            out = reshape_batch_dim_to_heads(self, out)
            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet, module_name=None):

        if module_name in ["attn1", "attn2"]:
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for k, net__ in net_.named_children():
                count = register_recr(net__, count, place_in_unet, module_name=k)
        return count

    cross_att_count = 0
    sub_nets = model.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count

def bw_hook(module, grad_input, grad_output):
    print('grad_output:', grad_output)

def fw_hook(module, input, output):
    W = module.to_out[0].weight
    b = module.to_out[0].bias
    reconstructed_input = torch.matmul((output - b).to(torch.float), torch.inverse(W.T.to(torch.float)))
    print(reconstructed_input.shape)
    print(output.shape)


def register_hook(net_, count, place_in_unet, module_name=None):
    if module_name in ["attn1", "attn2"]:

        net_.register_forward_hook(fw_hook)
        return count + 1
    elif hasattr(net_, 'children'):

        for k, net__ in net_.named_children():
            count = register_hook(net__, count, place_in_unet, module_name=k)
    return count

def save_tensor_as_image(image, path, plot=False):
    saved = image.clone()
    saved = (saved / 2 + 0.5).clamp(0, 1)
    saved = saved.detach().cpu().numpy()
    saved = (saved * 255).astype(np.uint8)
    if plot:
        plt.imshow(saved)
        plt.savefig(path)
    else:
        cv2.imwrite(path, saved)

def ddim_invert(unet, scheduler, latents, context, guidance_scale, num_inference_steps):
    
    timesteps = reversed(scheduler.timesteps)
    intermediate_latents = []
    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps-1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1: continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=context).sample

        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000//num_inference_steps)) #t
        next_t = t # t+1
        alpha_t = scheduler.alphas_cumprod[current_t]
        alpha_t_next = scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1-alpha_t).sqrt()*noise_pred)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noise_pred
        intermediate_latents.append(latents)
            
    return torch.cat(intermediate_latents)

def compute_embeddings(tokenizer, text_encoder, device, batch_size, prompt):
    text_input = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

    text_emb = text_encoder(text_input.input_ids.to(device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    context = torch.cat([uncond_embeddings, text_emb])
    return context