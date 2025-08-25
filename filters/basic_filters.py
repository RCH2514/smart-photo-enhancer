'''import cv2
import matplotlib.pyplot as plt

# Load your image
img = cv2.imread("im.jpg")  # place a test image in your repo folder

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Show original and grayscale
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")

plt.show()'''
import cv2
import matplotlib.pyplot as plt
from cv2 import dnn_superres

# Load the image
img_resized = cv2.imread("test.jpg")

# Resize to a smaller size (safe for memory)
#new_size = 512
#img_resized = cv2.resize(img, (new_size, new_size))

# Create super-resolution object
sr = dnn_superres.DnnSuperResImpl_create()
sr.readModel("FSRCNN_x3.pb")   # make sure this file is in the folder
sr.setModel("fsrcnn", 3)

# Upscale the resized image
img_upscaled = sr.upsample(img_resized)

# Convert BGR to RGB for Matplotlib
img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_upscaled_rgb = cv2.cvtColor(img_upscaled, cv2.COLOR_BGR2RGB)

# Display side by side
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img_resized_rgb)
plt.title("Resized Input")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(img_upscaled_rgb)
plt.title("AI Upscaled Output")
plt.axis("off")

plt.show()



