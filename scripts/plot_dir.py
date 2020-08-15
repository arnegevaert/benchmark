import sys
import json
from os import path
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from attrbench.evaluation.result import LinePlotResult


in_dir = sys.argv[1]
out_dir = sys.argv[2]
if not path.isdir(out_dir):
    os.makedirs(out_dir)

filenames = os.listdir(in_dir)
for filename in tqdm(filenames):
    out_filename = filename.split('.')[0] + '.png'
    with open(path.join(in_dir, filename), "r") as fp:
        data = json.load(fp)
    res = LinePlotResult(**data)
    fig, ax = res.plot(ci=True)
    fig.savefig(path.join(out_dir, out_filename), bbox_inches="tight")
    plt.close(fig)
