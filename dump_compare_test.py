import sys
import time
import cv2
from os.path import join, split
from os import listdir
import pickle
from vgg16 import vgg16

WORK_PATH = sys.argv[1]
VARIANT = sys.argv[2]
LAYER = sys.argv[3]
RESULTS_PATH = sys.argv[4]

renders_path = join(WORK_PATH, "renders", "ablation")
original_renders_paths = [join(renders_path, "original", f) for f in listdir(join(renders_path, "original"))]
variant_renders_paths = [join(renders_path, VARIANT, f) for f in listdir(join(renders_path, VARIANT))]

vgg16 = vgg16("vgg16_weights.npz")

layer = eval("vgg16.%s" % LAYER)

timestamp = time.strftime("%Y-%m-%d %H:%M")
results = {"time_start": timestamp, "work_path": WORK_PATH, "variant": VARIANT, "layer": LAYER, "results": {}}

for i in range(len(original_renders_paths)):

    print("%d/%d" % (i+1, len(original_renders_paths)))

    original_img_path = original_renders_paths[i]
    variant_img_path = variant_renders_paths[i]

    img_original = cv2.cvtColor(cv2.imread(original_img_path), cv2.COLOR_BGR2RGB)
    img_variant = cv2.cvtColor(cv2.imread(variant_img_path), cv2.COLOR_BGR2RGB)

    render_name = split(original_img_path)[-1]
    results["results"][render_name] = {"img_original": img_original, "img_variant": img_variant}

    _, layers_dump_original = vgg16.classify_with_dump(img_original, [layer])
    _, layers_dump_variant = vgg16.classify_with_dump(img_variant, [layer])

    results["results"][render_name]["layer_dump_original"] = layers_dump_original[0]
    results["results"][render_name]["layer_dump_variant"] = layers_dump_variant[0]

timestamp = time.strftime("%Y-%m-%d %H:%M")
results["time_end"] = timestamp
pickle.dump(results, open(RESULTS_PATH, "wb"))
