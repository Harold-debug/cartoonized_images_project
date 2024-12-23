"""
train_pix2pix.py

A simple Pix2Pix training script with hard-coded paths and parameters.

USAGE:
  python train_pix2pix.py

All default settings are defined in the code below.

Directory structure expected (from part 1):
  ./dataset/train/real/
  ./dataset/train/cartoon/
  ./dataset/test/real/
  ./dataset/test/cartoon/

Requirements:
  pip install tensorflow-macos tensorflow-metal opencv-python matplotlib
"""

import os
import glob
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from models import build_generator, build_discriminator

# ----------------------------
#    HARD-CODED CONSTANTS
# ----------------------------
TRAIN_REAL_DIR = "./dataset/train/real/"
TRAIN_CARTOON_DIR = "./dataset/train/cartoon/"
TEST_REAL_DIR = "./dataset/test/real/"
TEST_CARTOON_DIR = "./dataset/test/cartoon/"

CHECKPOINT_DIR = "./checkpoints_pix2pix/"
BEST_MODEL_DIR = "./best_model_pix2pix/"
SAMPLES_DIR = "./samples_pix2pix/"

IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 2
EPOCHS = 25
LEARNING_RATE = 2e-4  # Initial learning rate

SAVE_SAMPLES_EVERY = 5  # Save sample images every X epochs

# ----------------------------
#    EARLY STOPPING PARAMS
# ----------------------------
PATIENCE = 5          # Number of epochs with no improvement after which training will be stopped
MIN_DELTA = 0.0      # Minimum change in the monitored metric to qualify as an improvement

# ----------------------------
#       DATA LOADING
# ----------------------------
def load_image_pair(real_path, cartoon_path):
    """Load one (real, cartoon) image pair and preprocess."""
    # Read images
    real_img = cv2.imread(real_path)
    cartoon_img = cv2.imread(cartoon_path)

    # Convert BGR -> RGB
    real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
    cartoon_img = cv2.cvtColor(cartoon_img, cv2.COLOR_BGR2RGB)

    # Resize
    real_img = cv2.resize(real_img, (IMG_WIDTH, IMG_HEIGHT))
    cartoon_img = cv2.resize(cartoon_img, (IMG_WIDTH, IMG_HEIGHT))

    # Normalize from [0,255] to [-1,1]
    real_img = (real_img / 127.5) - 1.0
    cartoon_img = (cartoon_img / 127.5) - 1.0

    return real_img, cartoon_img

def build_dataset(real_dir, cartoon_dir):
    """Create a tf.data.Dataset yielding (real_image, cartoon_image) pairs."""
    real_paths = sorted(glob.glob(os.path.join(real_dir, "*.*")))
    cartoon_paths = sorted(glob.glob(os.path.join(cartoon_dir, "*.*")))

    assert len(real_paths) == len(cartoon_paths), (
        f"Mismatch: {len(real_paths)} real vs. {len(cartoon_paths)} cartoon images."
    )

    def generator():
        for r_path, c_path in zip(real_paths, cartoon_paths):
            real_img, cartoon_img = load_image_pair(r_path, c_path)
            yield real_img, cartoon_img

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=((IMG_HEIGHT, IMG_WIDTH, 3), (IMG_HEIGHT, IMG_WIDTH, 3))
    )

    # Shuffle and batch
    dataset = dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE)
    return dataset


# ----------------------------
#       LOSS FUNCTIONS
# ----------------------------
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    return 0.5 * (real_loss + generated_loss)

def generator_loss(disc_generated_output, gen_output, target):
    """Combined adversarial + L1 loss."""
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (100.0 * l1_loss)  # typical weighting
    return total_gen_loss, gan_loss, l1_loss

# ----------------------------
#    BUILD MODELS & OPTS
# ----------------------------
generator = build_generator()
discriminator = build_discriminator()

# ----------------------------
#   LEARNING RATE SCHEDULE
# ----------------------------
# Added: Exponential Decay Learning Rate Schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=LEARNING_RATE,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True
)

# Updated: Optimizers with Learning Rate Schedule
gen_optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5)

# ----------------------------
#        TRAIN STEP
# ----------------------------
@tf.function
def train_step(real_img, cartoon_img):
    """One training iteration."""
    with tf.GradientTape(persistent=True) as tape:
        # Generate output
        gen_output = generator(real_img, training=True)

        # Discriminator output
        disc_real = discriminator([real_img, cartoon_img], training=True)
        disc_generated = discriminator([real_img, gen_output], training=True)

        # Losses
        disc_loss = discriminator_loss(disc_real, disc_generated)
        gen_total_loss, gan_loss, l1_loss = generator_loss(disc_generated, gen_output, cartoon_img)

    # Gradients
    generator_grad = tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_grad = tape.gradient(disc_loss, discriminator.trainable_variables)

    # Update weights
    gen_optimizer.apply_gradients(zip(generator_grad, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(discriminator_grad, discriminator.trainable_variables))

    return gen_total_loss, disc_loss, gan_loss, l1_loss

# ----------------------------
#   CHECKPOINT MANAGEMENT
# ----------------------------
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")

ckpt = tf.train.Checkpoint(
    generator=generator,
    discriminator=discriminator,
    gen_optimizer=gen_optimizer,
    disc_optimizer=disc_optimizer
)

manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, max_to_keep=5)

# Best model checkpoint
best_ckpt = tf.train.Checkpoint(
    generator=generator
)
best_manager = tf.train.CheckpointManager(best_ckpt, BEST_MODEL_DIR, max_to_keep=1)

if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint)
    print(f"[INFO] Restored from {manager.latest_checkpoint}")
else:
    print("[INFO] Initializing from scratch...")

# ----------------------------
#   HELPER: SAVE SAMPLE IMG
# ----------------------------
def denormalize(x):
    """Convert [-1,1] back to [0,255]."""
    return (x + 1) * 127.5

def generate_and_save_images(model, test_input, real_target, epoch):
    """Generate sample and save to SAMPLES_DIR."""
    prediction = model(test_input, training=False)

    fig, axes = plt.subplots(1, 3, figsize=(12,4))

    # Input
    axes[0].imshow(denormalize(test_input[0]).numpy().astype(np.uint8))
    axes[0].set_title("Input (Real)")
    axes[0].axis("off")

    # Generated
    axes[1].imshow(denormalize(prediction[0]).numpy().astype(np.uint8))
    axes[1].set_title("Generated (Cartoon)")
    axes[1].axis("off")

    # Target
    axes[2].imshow(denormalize(real_target[0]).numpy().astype(np.uint8))
    axes[2].set_title("Target (Cartoon)")
    axes[2].axis("off")

    plt.tight_layout()

    os.makedirs(SAMPLES_DIR, exist_ok=True)
    save_path = os.path.join(SAMPLES_DIR, f"epoch_{epoch+1}.png")
    plt.savefig(save_path)
    plt.close()

# ----------------------------
#       TRAINING LOOP
# ----------------------------
def train(dataset, test_dataset, epochs):
    best_val_loss = np.Inf
    epochs_without_improvement = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        step = 0

        for real_img, cartoon_img in dataset:
            gen_loss, disc_loss, gan_loss, l1_loss = train_step(real_img, cartoon_img)

            # Print progress every 100 steps
            if step % 100 == 0:
                print(f"  Step {step} => Gen_loss: {gen_loss:.4f}, Disc_loss: {disc_loss:.4f}")
            step += 1

        # Save checkpoint each epoch
        manager.save()
        print(f"  [INFO] Checkpoint saved at epoch {epoch+1}.")

        # Save sample images every SAVE_SAMPLES_EVERY epochs
        if (epoch + 1) % SAVE_SAMPLES_EVERY == 0:
            for inp, tar in test_dataset.take(1):
                generate_and_save_images(generator, inp, tar, epoch)
            print(f"  [INFO] Sample images saved at epoch {epoch+1}.")

        # Early Stopping: Compute validation loss (average L1 loss on test dataset)
        val_l1_loss = compute_validation_loss(test_dataset)
        print(f"  Validation L1 Loss: {val_l1_loss:.4f}")

        # Check if validation loss improved
        if val_l1_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_l1_loss
            epochs_without_improvement = 0
            # Save the best model
            best_manager.save()
            print(f"  [INFO] Validation loss improved to {best_val_loss:.4f}. Best model saved.")
        else:
            epochs_without_improvement += 1
            print(f"  [INFO] No improvement in validation loss for {epochs_without_improvement} epoch(s).")

        # Check if patience is exceeded
        if epochs_without_improvement >= PATIENCE:
            print(f"  [INFO] Early stopping triggered after {epoch+1} epochs.")
            break

    print("[INFO] Training finished!")

def compute_validation_loss(test_dataset):
    """Compute the average L1 loss on the test dataset."""
    total_l1_loss = 0.0
    num_batches = 0

    for real_img, cartoon_img in test_dataset:
        # Generate output
        gen_output = generator(real_img, training=False)

        # Compute L1 loss
        l1_loss = tf.reduce_mean(tf.abs(cartoon_img - gen_output))
        total_l1_loss += l1_loss.numpy()
        num_batches += 1

    average_l1_loss = total_l1_loss / num_batches if num_batches > 0 else 0.0
    return average_l1_loss

# ----------------------------
#           MAIN
# ----------------------------
def main():
    # Build training & testing datasets
    train_ds = build_dataset(TRAIN_REAL_DIR, TRAIN_CARTOON_DIR)
    test_ds = build_dataset(TEST_REAL_DIR, TEST_CARTOON_DIR)

    print("[INFO] Starting Training...")
    train(train_ds, test_ds, EPOCHS)
    print("[INFO] Done.")

if __name__ == "__main__":
    main()