import keras_cv
import matplotlib.pyplot as plt
import time

# Start timer
start = time.time()

# Load Stable Diffusion model
print("Loading model...")

model = keras_cv.models.StableDiffusion(
    img_width=256,
    img_height=256
)

print("Model loaded successfully!")

# Generate image
print("Generating image... Please wait.")

images = model.text_to_image(
    "cute magical flying dog, fantasy art, golden color, high quality, highly detailed",
    batch_size=1
)

print("Image generated successfully!")

# Save image
plt.imshow(images[0].astype("uint8"))
plt.axis("off")

# Save to file
plt.savefig("generated_image.png", bbox_inches="tight", pad_inches=0)

# Show image
plt.show()

# End timer
end = time.time()

print(f"Image saved as generated_image.png")
print(f"Total time taken: {round(end - start, 2)} seconds")