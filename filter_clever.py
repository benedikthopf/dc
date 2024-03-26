from dataset import CLEVER
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

dataset = CLEVER("train", max_num_objects=6, resolution=(128, 128))
target_path = "/home/benedikthopf/Dokumente/9. Semester/datasets/CLEVER6"

for i, x in tqdm(enumerate(dataset)):
    img = x["image"]
    img = Image.fromarray(
        (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    img.save(os.path.join(target_path, f"{i}.png"))
