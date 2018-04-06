import os
import sys
import random
import pickle
import matplotlib.pyplot as plt
from utils import make_blended_heatmap

RESULTS_PATH = sys.argv[1]

walk = list(os.walk(RESULTS_PATH))

while True:
    p = random.choice(walk)
    files_list = p[2]
    if len(files_list) == 0:
        continue
    else:
        filename = random.choice(files_list)
        filepath = os.path.join(p[0], filename)
        data = pickle.load(open(filepath, "rb"))

        results = data["results"]
        image = data["image"]
        parameters, heatmap = random.choice(results)

        blended_heatmap = make_blended_heatmap(image, heatmap)

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(blended_heatmap)
        plt.title(str(parameters))
        plt.show()
