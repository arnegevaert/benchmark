import sys
import json
from os import path
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

import os
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from result import LinePlotResult, BoxPlotResult
from bam.model_contrast_score import MCSResult

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()


    lp_in_dir = path.join(args.in_dir, "lineplot")
    bp_in_dir = path.join(args.in_dir, "boxplot")
    lp_out_dir = path.join(args.out_dir, "lineplot")
    bp_out_dir = path.join(args.out_dir, "boxplot")

    os.makedirs(lp_out_dir)
    os.makedirs(bp_out_dir)

    filenames = os.listdir(lp_in_dir)
    for filename in tqdm(filenames):
        out_filename = filename.split('.')[0] + '.png'
        with open(path.join(lp_in_dir, filename), "r") as fp:
            data = json.load(fp)
        res = LinePlotResult(**data)
        fig, ax = res.plot(ci=True, title=filename)
        fig.savefig(path.join(lp_out_dir, out_filename), bbox_inches="tight")
        plt.close(fig)
    
    filenames = os.listdir(bp_in_dir)
    for filename in tqdm(filenames):
        out_filename = filename.split('.')[0] + '.png'
        with open(path.join(bp_in_dir, filename), "r") as fp:
            data = json.load(fp)
        res = BoxPlotResult(**data)
        fig, ax = res.plot(title=filename)
        fig.savefig(path.join(bp_out_dir, out_filename), bbox_inches="tight")
        plt.close(fig)

    with open(path.join(args.in_dir, "model_contrast_score.json"), "r") as fp:
        data = json.load(fp)
    res = MCSResult(**data)
    fig, ax = res.plot(title="model_contrast_score.json")
    fig.savefig(path.join(bp_out_dir, out_filename), bbox_inches="tight")
    plt.close(fig)
