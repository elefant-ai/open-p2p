import torch
import time
from elefant.data.rescale.resize import resize_image_for_model
from torchvision.transforms.functional import resize


def benchmark_image_rescale():
    im_in = torch.randint(0, 255, (3, 480, 640), dtype=torch.uint8)

    start = time.time()
    for i in range(1000):
        im_out = resize_image_for_model(im_in, (192, 192))
    end = time.time()
    print(f"Time taken rust: {end - start} seconds")

    start = time.time()
    for i in range(1000):
        im_out = resize(im_in, (192, 192))
    end = time.time()
    print(f"Time taken torchvision: {end - start} seconds")


def test_image_rescale():
    im_in = torch.randint(0, 255, (3, 480, 640), dtype=torch.uint8)
    im_out = resize_image_for_model(im_in, (192, 192))
    im_out_torch = resize(im_in, (192, 192))

    # Not sure what to do here, at least just check it runs.
