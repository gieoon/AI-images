import math
from inspect import isfunction
from functools import partial

%matplotlib inline
import matplotlib.pyplot as plt # Graphs
from tqdm.auto import tqdm #Progress bar
from einops import rearrange # Einstein operations for einstein formatting of tensor transformations.

import torch
from torch import nn, einsum
import torch.nn.functional as F

torch.manual_seed(0)

def exists(x):
    return x is not None

# Executes the function if it is one, otherwise, just returns the value
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Residual(nn.Module):
    
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

# Why does the network need to know what timestep it is dealing with?
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        helf_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        print("embeddings: ", embeddings.shape)
        # What are the values in these embeddings, how do they look?
        return embeddings
    
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        # Why is scaling done before SiLU?
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        # Some of the input flows directly through, ignoring block1 and block2, and is added to output.
        print("ResnetBlock output: ", (h + self.res_conv(x)).shape)
        return h + self.res_conv(x)

class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, time_emb_dim = None, mult = 2, norm = True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim_out, 7, padding = 3, groups = dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity()
            nn.Conv2d(dim, dim_out * mult, 3, padding = 1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding = 1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        h = self.ds_conv(x)

        if exists(self.mlp) and exists(time_emb):
            assert exists(time_emb), "time_emb must be passed in"
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)

# Attention layers added in between Conv2d blocks.
# qkv come from Conv2d and are trainable.
class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        # Query, Key, Value copies
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h = self.heads),
            qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach() # Don't use backpropagation on this.
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x = h, y = w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b c h (x y)", h = self.heads),
            qkv
        )
        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)
        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h = self.heads, x = h, y = w)
        return self.to_out(out)

# There is debate around whether to apply group normalization before or after attention in Transformers.
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

"""
Network
Takes in noisy images and noise levels per data sample, and outputs the predicted noise for each input.
Input: (batch_size, num_channels, height, width) + (batch_size, 1)
Output: (batch_size, num_channels, height, width)
"""
class Unet(nn.Module):
    def __init__(
        self, 
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1,2,4,8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
    ):

        super().__init__()
        
        # Determine dimensions
        self.channels = channels

        # Return init_dim if it exists, otherwise the second parameter.
        init_dim = default(init_dim, dim // 3 * 2)

        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding = 3)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[: - 1], dims[1:]))

        if use_convnext:
            block_klass = partial(ConvNextBlock, mult = convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # Time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None
        
        # Layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim = time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim = time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim = time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)

        self.final_conv = nn.Sequential(
            block_klass(dim, dim), 
            nn.Conv2d(dim, dim_out, 1)
        )
    
    def forward(self, x, time):
        
        print("Unet forward shape: ", x.shape)

        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn, _downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = _downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, _upsample, in self.ups:
            # Concatenate x and the last attention output
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = _upsample(x)
        
        return self.final_conv(x)

# Variance schedule for noise.
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

TIMESTEPS = 200

betas = linear_beta_schedule(timesteps=TIMESTEPS)

alphas = 1. - betas # 1 - beta schedule variance
alphas_cumprod = torch.cumprod(alphas, axis=0) # alphabar, cumulative product of all the variances.
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# Calculations for diffusion q(x_t | x_{t-1})
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod) 
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# Calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# Extract the timestep t indices for a batch of indices.
def extract_t(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
test_image = Image.open(requests.get(url, stream=True).raw)
print("test_image: ", test_image)

from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

image_size = 128
# Process from rgb 255 values into linearly scaled values [-1, 1]
transform = Compose([
    Resize(image_size),
    CenterCrop(image_size),
    ToTensor(),
    Lambda(lambda t: (t * 2) - 1)
])

x_start = transform(test_image).unsqueeze(0)
print("x_start.shape: ", x_start.shape)

# PIL image back to rgb 255
reverse_transform = Compose([
    Lambda(lambda t: (t + 1) / 2),
    Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
    Lambda(lambda t: t * 255),
    Lambda(lambda t: t.numpy().astype(np.uint8)),
    ToPILImage(),
])

reverse_transform(x_start.squeeze())
    
# Forward diffusion (Using sum of gaussian distribution is another gaussian theorem.)
def q_sample(x_start, t, noise=None):
    if noise == None:
        noise = torch.randn_like(x_start)
    
    sqrt_alphas_cumprod_t = extract_t(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract_t(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def test_noisy_image(x_start, t):

    x_noisy = q_sample(x_start, t=t)
    
    noisy_image = reverse_transform(x_noisy.squeeze())

    return noisy_image

test_timestep = torch.tensor([40])
test_noisy_image(test_image, test_timestep)

def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(figsize=(200,200), nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()

plot([test_noisy_image(x_start, torch.tensor([t])) for t in [0, 50, 100, 150, 199]])

def p_losses(denoise_model, x_start, t, noise=None, loss_type='l1'):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == 'huber':
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

from datasets import load_dataset

dataset = load_dataset("fashion_mnist")
image_size = 28
channels = 1
batch_size = 128

from torchvision import transforms
from torch.utils.data import DataLoader

transform = Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
])

def transforms(examples):
   examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
   del examples["image"]

   return examples

transformed_dataset = dataset.with_transform(transforms).remove_columns("label")

dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)

batch = next(iter(dataloader))
print(batch.keys())

@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract_t(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract_t(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract_t(
        sqrt_alphas_cumprod, t, x.shape
    )

    # Equation 11 in the paper
    # Use the NN model to predict the mean.
    # Model output is noise, conditioned on x and timestep.
    # model_mean = sqrt_recip_alphas_t * (x - betas_t / sqrt_one_minus_alphas_cumprod_t * model(x, t))
    # Not sure if this is the same as above, but don't want to risk it.
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    print("model_mean: ", model_mean.shape)
    print("posterior_variance: ", posterior_variance.shape)
    print("posterior_variance_t: ", posterior_variance_t.shape)

    if t_index == 0:
        # This is the final distribution
        return model_mean
    else:
        # Variance schedule for x{t-1} given x{t} and x0
        # posterior_variance is calculated as a fraction of the previous timestep compared to this timestep.
        posterior_variance_t = extract_t(
            posterior_variance, t, x.shape
        )

        # Create new noise here.
        noise = torch.randn_like(x) # z ~ N(0, I)  # Sample noise from isotropic normal/gaussian distribution

        # Algorithm 2, line 4.
        # Take the original mean and multiply by noise, but with less variance, since posterior_variance is a fraction.
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]

    # Start from pure noise for each image in the batch.
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, TIMESTEPS)), desc='sampling loop time step', total=TIMESTEPS):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    
    return imgs

@torch.no_grad
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


from pathlib import Path

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    
    return arr

results_folder = Path("./train_results")
results_folder.mkdir(exist_ok = True)
save_and_sample_every = 1000

from torch.optim import Adam

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4)
)
model.to(device)

LEARNING_RATE = 1e-3
EPOCHS = 5

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

from torchvision.utils import save_image

def train():
    for epoch in range(EPOCHS):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch_size = batch['pixel_values'].shape[0]
            batch = batch['pixel_values'].to(device)

            # Algorithm 1 line 3: sample t uniformly for every example in the batch
            timestep = torch.randint(0, TIMESTEPS, (batch_size, ), device=device).long() #torch.randn_like(batch.shape)

            loss = p_loss(model, batch, timestep, noise=None, loss_type='huber')

            if step % 100 == 0:
                print("Loss: ", loss.item())

            loss.backward()
            optimizer.step()

            # Save generated images
            if step != 0 and step % save_and_sample_every == 0:
                milestone = step // save_and_sample_every
                batches = num_to_groups(4, batch_size)
                all_images_list = list(map(lambda n: sample(model, batch_size=n, channels=channels), batches))
                all_images = torch.cat(all_images_list, dim=0)
                all_images = (all_images + 1) * 0.5
                save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6)
            
# Inference
samples = sample(model, image_size=image_size, batch_size=64, channels=channels)

random_index = 5
plt.imshow(samples[-1][random_index].reshape(image_size, image_size, channels), cmap='gray')

# Create gif of the denoising process
import matplotlib.animation as animation

def create_denoising_gif():
    random_index = 53

    fig = plt.figure()
    ims = []
    for i in range(timesteps):
        im = plt.imshow(samples[i][random_index].reshape(image_size, image_size, channels), cmap="gray", animated=True)
        ims.append([im])

    animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    animate.save('diffusion.gif')
    plt.show()

