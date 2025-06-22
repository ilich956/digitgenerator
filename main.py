import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_size, channels):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.channels = channels
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, channels * img_size * img_size),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        c = self.label_embedding(labels)
        gen_input = torch.cat((noise, c), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), self.channels, self.img_size, self.img_size)
        return img

LATENT_DIM = 100
NUM_CLASSES = 10
IMG_SIZE = 28
CHANNELS = 1
MODEL_PATH = "digit_generator_cgan.pth" # Path to your trained model file

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource # Cache the model loading to prevent reloading on every rerun
def load_generator_model():
    """Loads the pre-trained generator model."""
    try:
        generator = Generator(LATENT_DIM, NUM_CLASSES, IMG_SIZE, CHANNELS).to(DEVICE)
        generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        generator.eval() # Set to evaluation mode for inference
        st.success(f"Model loaded successfully on {DEVICE}!")
        return generator
    except FileNotFoundError:
        st.error(f"Error: Model file '{MODEL_PATH}' not found. Please ensure it's in the correct directory.")
        st.stop() 
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()

# Load the model once when the app starts
generator = load_generator_model()

# --- 3. Image Generation Function ---
def generate_images(generator_model, target_digit, num_images=5):
    """Generates a specified number of images for a given digit."""
    # Generate unique noise vectors for diversity
    noise_vectors = torch.randn(num_images, LATENT_DIM, device=DEVICE)
    
    # Create label tensor for the target digit
    target_labels = torch.full((num_images,), target_digit, dtype=torch.long, device=DEVICE)

    with torch.no_grad(): # Disable gradient calculations for inference
        generated_tensors = generator_model(noise_vectors, target_labels).cpu()

    # Denormalize images from [-1, 1] to [0, 1] for display
    generated_images = (generated_tensors + 1) / 2

    # Convert tensors to PIL Images
    pil_images = [transforms.ToPILImage()(img) for img in generated_images]
    return pil_images

# --- 4. Streamlit Web Application Layout ---
st.set_page_config(
    page_title="Handwritten Digit Generator",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Handwritten Digit Generator")
st.markdown("Select a digit (0-9) and click 'Generate' to see 5 unique handwritten images!")

# Sidebar for digit selection
st.sidebar.header("Settings")
selected_digit = st.sidebar.slider(
    "Choose a digit to generate:",
    min_value=0,
    max_value=9,
    value=0,
    step=1
)

# Button to trigger generation
if st.sidebar.button("Generate Digit Images"):
    st.subheader(f"Generating 5 images for digit {selected_digit}...")
    
    with st.spinner("Generating images... Please wait."):
        images = generate_images(generator, selected_digit, num_images=5)
    
    st.success("Images generated!")

    # Display the generated images in columns
    cols = st.columns(5)
    for i, img in enumerate(images):
        with cols[i]:
            st.image(img, caption=f"Generated {selected_digit}", use_container_width=True)


# Optional: Add a section for model info (e.g., if you want to display training details)
# st.expander("Model Details").write("Your model details here...")
