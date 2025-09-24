from typing import List, Union
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from PIL import Image
import os
import io
from transformer import Transformer
from vq_vae import VQVAE
import numpy


type_size = 50
colors_size = 39


from fastapi import FastAPI

app = FastAPI()


app = FastAPI(title="Model generation API")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 64

class GenerateResponse(BaseModel):
    text: str
    metadata: dict = {}
    
class GenerationParams(BaseModel):
    is_block: int
    type_idx: List[int]
    color_idx: List[int]
    temperature: float


TF_GENERATOR_PATH = os.environ.get("MODEL_PATH", "transformer.pth")
VAE_NET_PATH = os.environ.get("MODEL_PATH", "vae_net.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model on startup
async def load_model():
    global tf_generator
    global vae_net
    try:
        vae_net = VQVAE(channel_in=4, latent_channels=256, ch=128,
                code_book_size=512, commitment_cost=0.50,
                cond_type_dim=type_size, cond_colors_dim=colors_size, cond_hidden=256).to(DEVICE)
        tf_generator = Transformer(
        num_emb=512 + 1,
        num_layers=6,
        hidden_size=256,
        num_heads=8,
        latent_channels=256,
        cond_type_dim=type_size,
        cond_colors_dim=colors_size).to(DEVICE)

        tf_generator.load_state_dict(torch.load(TF_GENERATOR_PATH, map_location=DEVICE))
        vae_net.load_state_dict(torch.load(VAE_NET_PATH,map_location=DEVICE))
        tf_generator.eval()
        vae_net.eval()
        return tf_generator, vae_net 
    except Exception as e:
        # crash early if model can't be loaded
        raise RuntimeError(f"Failed to load model: {e}")

app.add_event_handler("startup", load_model)

@app.post("/generate")
def generate(params: GenerationParams):
    '''
    Example input:
    {
    "is_block":1,
    "type_idx":[2],
    "color_idx":[33],
    "temperature":1.2
    }
    '''
    img = generate_example_image(
        is_block=params.is_block,
        type_idx=params.type_idx,
        color_idx=params.color_idx,
        temperature=params.temperature,)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


def tensor_to_image(tensor):
    """Convert tensor to PIL Image"""
    # Normalize
    tensor = (tensor + 1) / 2

    # Remove batch dimension and convert to CPU
    tensor = tensor.squeeze(0).cpu()

    tensor = tensor.permute(1, 2, 0)

    # Convert to 0-255 range
    tensor = (tensor * 255).byte()

    print(tensor.shape)

    # Convert to PIL Image
    image = Image.fromarray(tensor.numpy(), 'RGBA')

    return image

def generate_image(transformer, vae_net, is_block, type_, colors, latent_grid_size=8, temperature=1.5):
    transformer.eval()
    vae_net.eval()

    device = next(transformer.parameters()).device

    # Ensure proper conditioning dimensions
    is_block = is_block.to(device)  # Add batch dimension
    type_ = type_.to(device).unsqueeze(0)
    colors = colors.to(device).unsqueeze(0)

    # Start with SOS token
    current_tokens = torch.zeros(1, 1, dtype=torch.long, device=device)

    with torch.no_grad():
        for i in range(latent_grid_size * latent_grid_size):
            logits = transformer(current_tokens, is_block=is_block, type_=type_, colors=colors)
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            current_tokens = torch.cat([current_tokens, next_token], dim=1)

    # Remove SOS token and ensure valid token range
    tokens = current_tokens[:, 1:]
    tokens = torch.clamp(tokens, 0, vae_net.vq.code_book_size - 1)
    tokens = tokens.view(1, latent_grid_size, latent_grid_size)

    # Convert tokens to quantized representation
    quantized = vae_net.vq.embedding(tokens).permute(0, 3, 1, 2)

    # Decode with conditions
    generated_image = vae_net.decode(quantized, is_block=is_block, type_=type_, colors=colors)

    return generated_image

# Helper function to create condition tensors with correct dimensions
def create_conditions(is_block_val, type_idx, color_idx):
    """
    Create condition tensors with proper dimensions

    Args:
        is_block_val: 0 or 1
        [type_idx]: Indexes of type (0-50)
        [color_idx]: Indexes of color (0-41)

    Returns:
        is_block, type_, colors tensors
    """
    # Create is_block tensor
    is_block = torch.tensor([is_block_val], dtype=torch.float32)

    # Create one-hot type tensor
    type_ = torch.zeros(type_size)
    for i in type_idx:
        type_[i] = 1

    # Create one-hot colors tensor
    colors = torch.zeros(colors_size)
    for i in color_idx:
        colors[i] = 1

    return is_block, type_, colors

# Example usage:
def generate_example_image(is_block,type_idx,color_idx,temperature,tf_generator,vae_net):
    # Create conditions with proper dimensions
    is_block, type_, colors = create_conditions(
        is_block_val=is_block,      # is_block
        type_idx=type_idx,          # Type
        color_idx=color_idx          # Color
    )

    # Generate image
    generated_image = generate_image(
        transformer=tf_generator,
        vae_net=vae_net,
        is_block=is_block,
        type_=type_,
        colors=colors,
        latent_grid_size=4,
        temperature=temperature
    )

    # Convert to PIL Image and save
    image = tensor_to_image(generated_image)
    return image

