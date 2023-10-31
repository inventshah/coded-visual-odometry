import argparse
import os
import sys


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm

from configopts import get_experiment, DatasetName
from data import ICLDataset


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


parser = argparse.ArgumentParser()
parser.add_argument("--CONFIG", "-c", type=str, required=True)
parser.add_argument(
    "--DATASET",
    "-d",
    type=str,
    default="datasets",
    required=False,
)
parser.add_argument(
    "--CHECKPOINT",
    "-s",
    type=str,
    required=True,
)


args = parser.parse_args(sys.argv[1:])
DATASET_DIR = args.DATASET

EXPERIMENT = get_experiment(args.CONFIG)


# model

model = EXPERIMENT.model().to(device).eval()
model.load_state_dict(torch.load(args.CHECKPOINT, map_location=device))


print(repr(EXPERIMENT))


for name in DatasetName:
    dataset = ICLDataset(DATASET_DIR, name.name, EXPERIMENT.coded, cache=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, batch in tqdm.tqdm(
            enumerate(dataloader), total=len(dataset), desc=name.name
        ):
            recon = EXPERIMENT.post_forward(model(batch["Coded"].to(device)))
            metric_gt = batch["Depth"].to(device)

            metric_re = EXPERIMENT.depth.output_to_metric(recon).detach().cpu().numpy()

            np.save(
                os.path.join(DATASET_DIR, name.name, "Results", f"{i:04d}.npy"),
                metric_re,
            )
