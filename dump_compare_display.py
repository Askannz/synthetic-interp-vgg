import sys
import numpy as np
import cv2
import pickle


def get_heatmap_dims(data):

    n = int(np.log2(data.size))
    h = 2 ** (int(n/2))
    w = 2 ** (n - int(n/2))

    return h, w


class Manager:

    def __init__(self, diffs):

        self.diffs = diffs
        h, w = get_heatmap_dims(diffs[0])
        self.heatmap = np.zeros((h, w, 3), np.uint8)

        self.min_val = np.min(diffs)
        self.max_val = np.max(diffs)

    def make_heatmap(self, diff):

        h, w, _ = self.heatmap.shape
        diff = diff.reshape(h, w)
        norm_diff = (diff - self.min_val) / (self.max_val - self.min_val)

        self.heatmap[:, :, 0] = (norm_diff * 180).astype(np.uint8)
        self.heatmap[:, :, 1] = 255
        self.heatmap[:, :, 2] = 255

        self.heatmap = cv2.cvtColor(self.heatmap, cv2.COLOR_HSV2BGR)

    def get_disp(self):

        disp = cv2.resize(self.heatmap, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        return disp

    def slider_update(self, pos):

        self.make_heatmap(self.diffs[pos])


RESULTS_PATH = sys.argv[1]
WNAME = "Heatmaps"

results = pickle.load(open(RESULTS_PATH, "rb"))

diffs = []
for render_name in results["results"].keys():
    layer_dump_original = results["results"][render_name]["layer_dump_original"]
    layer_dump_variant = results["results"][render_name]["layer_dump_variant"]

    diff = np.abs(layer_dump_variant - layer_dump_original)
    diffs.append(diff)

h, w = get_heatmap_dims(diffs[0])
min_val, max_val = np.min(diffs), np.max(diffs)

cv2.namedWindow(WNAME)
manager = Manager(diffs)
cv2.createTrackbar("Render", WNAME, 0, len(diffs) - 1, manager.slider_update)

quit = False
while not quit:

    cv2.imshow(WNAME, manager.get_disp())

    key = cv2.waitKey(1)
    if key == ord('\x1b'):
        quit = True

cv2.destroyAllWindows()
