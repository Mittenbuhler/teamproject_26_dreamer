import gymnasium as gym
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def get_frame(env, resolution=32):
    frame = env.render()
    img = Image.fromarray(frame)
    img = img.convert("L")
    img = img.resize((resolution, resolution))
    return np.array(img, dtype=np.float32) / 255.0

env = gym.make("CartPole-v1", render_mode="rgb_array")
obs, _ = env.reset()

img = get_frame(env)
plt.imshow(img, cmap="gray")
plt.show()
