import argparse

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

input_data = np.load(args.input, allow_pickle=True)
with open(args.output, "wb") as f:
    f.write(input_data.astype(np.float32).tobytes())
