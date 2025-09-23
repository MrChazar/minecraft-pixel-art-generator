import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import os
import io

# import twoich klas
from transformer import Transformer
from vq_vae import VQVAE


# ================== KONFIGURACJA ==================
TF_GENERATOR_PATH = os.environ.get("MODEL_PATH", "transformer.pth")
VAE_NET_PATH = os.environ.get("MODEL_PATH", "vae_net.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================== FUNKCJE ==================
@st.cache_resource(show_spinner=True)
def load_model():
    """Ładowanie modeli tylko raz (cache'owane przez Streamlit)."""
    vae_net = VQVAE(
        channel_in=4,
        latent_channels=256,
        ch=128,
        code_book_size=512,
        commitment_cost=0.50,
        cond_type_dim=51,
        cond_colors_dim=42,
        cond_hidden=256
    ).to(DEVICE)

    tf_generator = Transformer(
        num_emb=512 + 1,
        num_layers=6,
        hidden_size=256,
        num_heads=8,
        latent_channels=256,
        cond_type_dim=51,
        cond_colors_dim=42
    ).to(DEVICE)

    tf_generator.load_state_dict(torch.load(TF_GENERATOR_PATH, map_location=DEVICE))
    vae_net.load_state_dict(torch.load(VAE_NET_PATH, map_location=DEVICE))
    tf_generator.eval()
    vae_net.eval()

    return tf_generator, vae_net


def tensor_to_image(tensor):
    """Konwersja tensora do obrazu PIL (RGBA)."""
    tensor = (tensor + 1) / 2
    tensor = tensor.squeeze(0).cpu()
    tensor = tensor.permute(1, 2, 0)
    tensor = (tensor * 255).byte()
    image = Image.fromarray(tensor.numpy(), 'RGBA')
    return image


def create_conditions(is_block_val, type_idx, color_idx):
    """Budowanie tensorów warunków tak jak w API."""
    is_block = torch.tensor([is_block_val], dtype=torch.float32)

    type_ = torch.zeros(51)
    for i in type_idx:
        type_[i] = 1

    colors = torch.zeros(42)
    for i in color_idx:
        colors[i] = 1

    return is_block, type_, colors


def generate_image(transformer, vae_net, is_block, type_, colors,
                   latent_grid_size=8, temperature=1.5):
    transformer.eval()
    vae_net.eval()
    device = next(transformer.parameters()).device

    is_block = is_block.to(device)
    type_ = type_.to(device).unsqueeze(0)
    colors = colors.to(device).unsqueeze(0)

    current_tokens = torch.zeros(1, 1, dtype=torch.long, device=device)

    with torch.no_grad():
        for i in range(latent_grid_size * latent_grid_size):
            logits = transformer(current_tokens, is_block=is_block, type_=type_, colors=colors)
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            current_tokens = torch.cat([current_tokens, next_token], dim=1)

    tokens = current_tokens[:, 1:]
    tokens = torch.clamp(tokens, 0, vae_net.vq.code_book_size - 1)
    tokens = tokens.view(1, latent_grid_size, latent_grid_size)

    quantized = vae_net.vq.embedding(tokens).permute(0, 3, 1, 2)
    generated_image = vae_net.decode(quantized, is_block=is_block, type_=type_, colors=colors)
    return generated_image


def generate_example_image(tf_generator, vae_net, is_block, type_idx, color_idx, temperature):
    is_block, type_, colors = create_conditions(
        is_block_val=is_block,
        type_idx=type_idx,
        color_idx=color_idx
    )
    generated_tensor = generate_image(
        transformer=tf_generator,
        vae_net=vae_net,
        is_block=is_block,
        type_=type_,
        colors=colors,
        latent_grid_size=4,
        temperature=temperature
    )
    image = tensor_to_image(generated_tensor)
    return image


# ================== INTERFEJS STREAMLIT ==================
st.set_page_config(page_title="Generator Obrazków", page_icon="🎨", layout="centered")
st.title("🎨 Generator obrazków (Streamlit + VQ-VAE + Transformer)")

# ładowanie modeli
with st.spinner("Ładowanie modeli..."):
    tf_generator, vae_net = load_model()

# pola wejściowe
col1, col2, col3 = st.columns(3)
with col1:
    type_idx = st.text_input("Type index (np. 2,5,10)", "2")
with col2:
    color_idx = st.text_input("Color index (np. 33,12)", "33")
with col3:
    is_block = st.selectbox("Is block?", [0, 1], index=0)

temperature = st.slider("Temperature", 0.5, 2.0, 1.2, 0.1)

if st.button("Generuj obrazek"):
    try:
        type_list = [int(x.strip()) for x in type_idx.split(",") if x.strip().isdigit()]
        color_list = [int(x.strip()) for x in color_idx.split(",") if x.strip().isdigit()]
        is_block_val = int(is_block)

        img = generate_example_image(
            tf_generator=tf_generator,
            vae_net=vae_net,
            is_block=is_block_val,
            type_idx=type_list,
            color_idx=color_list,
            temperature=temperature
        )

        img = img.resize((16, 16), resample=Image.NEAREST)
        st.image(img, caption="Wersja 16x16", use_column_width=False)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.download_button(
            label="Pobierz obrazek",
            data=buf.getvalue(),
            file_name="generated.png",
            mime="image/png"
        )

    except Exception as e:
        st.error(f"Coś się wywaliło: {e}")
