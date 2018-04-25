import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import make_blended_heatmap

RESULTS_PATH = sys.argv[1]

walk = list(os.walk(RESULTS_PATH))

for p in walk:
    files_list = p[2]
    if len(files_list) == 0:
        continue
    else:
        for filename in files_list:
            filepath = os.path.join(p[0], filename)
            data = pickle.load(open(filepath, "rb"))

            results = data["results"]
            image = data["image"]

            ordered_results = {"disc": {}, "disc_grad": {}, "square": {}}

            for parameters, heatmap in results:

                shape = parameters["shape"]
                size = parameters["size"]
                color = parameters["color"]
                if color != [128, 128, 128]:
                    continue
                ordered_results[shape][size] = heatmap

            sizes = ordered_results["disc"].keys()
            shapes = ordered_results.keys()

            max_diff = -float("inf")
            for j, size in enumerate(sorted(sizes)):
                for i, shape in enumerate(shapes):
                    diff = np.max(ordered_results[shape][size])
                    if diff > max_diff:
                        max_diff = diff

            plt.figure()
            for j, size in enumerate(sorted(sizes)):
                for i, shape in enumerate(shapes):
                    heatmap = ordered_results[shape][size]
                    blended_heatmap = make_blended_heatmap(image, heatmap, max_diff)
                    plt.subplot(len(sizes), len(shapes), j * len(shapes) + i + 1)
                    plt.imshow(blended_heatmap)
                    shape_strs = {"disc": "Disc", "square": "Square", "disc_grad": "Gradient"}
                    plt.title("%s, %dpx" % (shape_strs[shape], size))
                    plt.xticks([])
                    plt.yticks([])
            plt.tight_layout()
            """plt.figure()
            plt.imshow(image)"""
            plt.show()
