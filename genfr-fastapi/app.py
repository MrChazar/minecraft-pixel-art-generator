import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import os
import io
import asyncio

# import twoich klas
from transformer import Transformer
from vq_vae import VQVAE
from main import load_model, generate_example_image
from mapping import types, colors


# ================== KONFIGURACJA ==================
TF_GENERATOR_PATH = os.environ.get("MODEL_PATH", "transformer.pth")
VAE_NET_PATH = os.environ.get("MODEL_PATH", "vae_net.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== INTERFEJS STREAMLIT ==================
st.set_page_config(page_title="GENERATOR FRAJDY", page_icon="🎨", layout="centered")
st.title("GENERATOR FRAJDY")

# ładowanie modeli
with st.spinner("Loading models..."):
    tf_generator, vae_net = asyncio.run(load_model())
# pola wejściowe
col1, col2, col3 = st.columns(3)
with col1:
    type_idx = st.multiselect('Select types', types.keys())
with col2:
    color_idx = st.multiselect('Select colors', colors.keys())
with col3:
    is_block_str = st.selectbox("Is block?", ["No", "Yes"])
    is_block_val = 1 if "Yes" in is_block_str else 0 

temperature = st.slider("Temperature", 0.01, 2.0, 1.2, 0.1)

if st.button("Generate image"):
    try:
        type_list = [types[name] for name in type_idx]
        color_list = [colors[name] for name in color_idx]
        is_block_val = int(is_block_val)
    
        img = generate_example_image(
            tf_generator=tf_generator,
            vae_net=vae_net,
            is_block=is_block_val,
            type_idx=type_list,
            color_idx=color_list,
            temperature=temperature
        )

        img = img.resize((256, 256), resample=Image.NEAREST)
        st.image(img, caption="Generated", use_container_width=False)
        
        img = img.resize((16, 16), resample=Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.download_button(
            label="Download image",
            data=buf.getvalue(),
            file_name="generated.png",
            mime="image/png"
        )

    except Exception as e:
        st.error(f"Something went wrong: {e}")
