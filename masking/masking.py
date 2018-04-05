import sys
from os import listdir
from os.path import join, splitext
import time
import json
import pickle
import cv2
from masking_processor import MaskingProcessor

config = json.load(open(sys.argv[1]))

RESULTS_BASE_PATH = sys.argv[2]

BASE_PATH = config["base_path"]
OBJECTS_PATHS = config["objects_paths"]
CLASSES_IDS = config["classes_ids"]
SIZES = config["sizes"]
STRIDE = int(config["stride"])
COLORS = config["colors"]
SHAPES = "square", "disc", "disc_grad"
MONITOR = bool(config["monitor"])

processor = MaskingProcessor(monitor=MONITOR)

for i, object_path in enumerate(OBJECTS_PATHS):

    image_base_path = join(BASE_PATH, object_path)
    files = listdir(image_base_path)

    j = 0
    for f in files:
        if splitext(f)[1] == '.jpg':

            path = join(image_base_path, f)
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            classes_ids = CLASSES_IDS[i]

            print("Testing %d/%d of object %s" % (j+1, len(files), object_path))

            timestamp = time.strftime("%Y-%m-%d %H:%M")
            results = {"time_start": timestamp, "path": path, "image": img, "classes_ids": classes_ids, "results": []}
            for s in SIZES:
                for color in COLORS:
                    for shape in SHAPES:
                        print(s, color, shape)
                        parameters = {"size": s, "stride": STRIDE, "color": color, "shape": shape}
                        heatmap = processor.process(img, classes_ids, parameters)
                        results["results"].append((parameters, heatmap))
            results_path = join(RESULTS_BASE_PATH, object_path, str(j+1))
            timestamp = time.strftime("%Y-%m-%d %H:%M")
            results["time_end"] = timestamp
            pickle.dump(results, open(results_path, "wb"))
            j += 1
