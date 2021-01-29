from attrbench.lib.masking import ConstantMasker, BlurringMasker, SampleAverageMasker
from attrbench.test.util import generate_images_indices
import time


kernel_size = 3
blurring = BlurringMasker("channel", kernel_size)
cst = ConstantMasker("channel")
mask_size = 500
shape = (16, 1, 100, 100)
n = 100
images, indices = generate_images_indices(shape, mask_size, "pixel")

maskers = {
    "blur": BlurringMasker("channel", kernel_size),
    "avg": SampleAverageMasker("channel"),
    "cst": ConstantMasker("channel")
}

for name in maskers:
    masker = maskers[name]
    start_t = time.time()
    for i in range(n):
        masker.mask(images, indices)
    end_t = time.time()
    print(f"{name}: {end_t - start_t}s")
