import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image
from models import build_generator

# ----------------------------
#    DIRECTORY PATHS
# ----------------------------
BEST_MODEL_DIR = "./best_model_pix2pix/"

# ----------------------------
#   GENERATOR LOADING FUNCTION
# ----------------------------
@st.cache_resource
def load_generator():
    """Load the generator model from the best checkpoint."""
    generator = build_generator()
    best_ckpt = tf.train.Checkpoint(generator=generator)
    best_manager = tf.train.CheckpointManager(best_ckpt, BEST_MODEL_DIR, max_to_keep=1)

    if best_manager.latest_checkpoint:
        best_ckpt.restore(best_manager.latest_checkpoint).expect_partial()
        st.write(f"🎉 [INFO] Restored best model from `{best_manager.latest_checkpoint}`!")
    else:
        st.write("🚨 [ERROR] No best model checkpoint found. Please train the model first.")
    return generator

# Load the generator
generator = load_generator()

# ----------------------------
#        STREAMLIT UI
# ----------------------------
st.title("✨ Image Cartoonization with Pix2Pix GAN ✨")
st.markdown("Upload a real-world image, and get its cartoonized version in no time!")

# Upload image
uploaded_file = st.file_uploader("📤 Upload a real image...", type=["jpg", "jpeg", "png", "bmp"])

# Show the uploaded image and process
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.subheader("🖼️ Uploaded Image")

    # Preprocess the image
    def preprocess_image(image):
        """Preprocess the uploaded image for model inference."""
        image = image.resize((256, 256))
        image = np.array(image)
        image = (image / 127.5) - 1.0  # Normalize to [-1, 1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image.astype(np.float32)

    input_image = preprocess_image(image)

    # Cartoonize button
    if st.button("✨ Cartoonize! 🎨"):
        if generator is not None:
            with st.spinner("🖌️ Generating cartoonized image..."):
                # Generate cartoonized image
                generated_image = generator(input_image, training=False)
                generated_image = (generated_image[0].numpy() + 1.0) * 127.5  # Denormalize to [0,255]
                generated_image = generated_image.astype(np.uint8)

                st.subheader("📸 Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original Image", use_container_width=True)
                with col2:
                    st.image(generated_image, caption="Cartoonized Image", use_container_width=True)
        else:
            st.error("🚨 Generator model not loaded. Please check your setup.")
else:
    st.info("👆 Upload an image to begin!")