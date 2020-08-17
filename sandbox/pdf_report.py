import sys
from os import path
from tqdm import tqdm
from fpdf import FPDF


in_dir = sys.argv[1]
out_file = sys.argv[2]


templates = []
for ds in ["MNIST", "CIFAR10", "ImageNette"]:
    for metric in ["insertion", "deletion", "infidelity", "max_sensitivity", "sensitivity_n"]:
        templates.append(f"{ds}_{metric}_{{}}.png")

y = 0
pdf = FPDF()
for template in tqdm(templates):
    if y == 0:
        pdf.add_page()
    ident_fn = path.join(in_dir, template.format("identity"))
    softmax_fn = path.join(in_dir, template.format("softmax"))
    pdf.image(ident_fn, x=0, y=y, w=100)
    pdf.image(softmax_fn, x=100, y=y, w=100)
    y = (y + 100) % 300
pdf.output(out_file)
