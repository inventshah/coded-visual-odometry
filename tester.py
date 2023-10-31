import argparse
import os
import sys


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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


# datasets

icl_datasets = {
    name: ICLDataset(DATASET_DIR, name.name, EXPERIMENT.coded, cache=False)
    for name in DatasetName
}

test_loaders = {
    name: DataLoader(dataset, batch_size=EXPERIMENT.batch_size, shuffle=True)
    for name, dataset in icl_datasets.items()
}

# model

model = EXPERIMENT.model().to(device).eval()
model.load_state_dict(torch.load(args.CHECKPOINT, map_location=device))

L1 = nn.L1Loss()

print(repr(EXPERIMENT))


def to_numpy(img: torch.Tensor):
    return np.clip(img.detach().cpu().numpy(), 0, None)


def evaluate(dataloader: DataLoader):
    model.eval()
    with torch.no_grad():
        metric_depth_error = 0
        metric_depth_error_under3 = 0
        sample_count = 0

        for batch in dataloader:
            recon = EXPERIMENT.post_forward(model(batch["Coded"].to(device)))
            metric_gt = batch["Depth"].to(device)

            metric_re = EXPERIMENT.depth.output_to_metric(recon)

            mask = metric_gt < 3
            if torch.any(mask).item():
                metric_depth_error_under3 += L1(
                    metric_re[mask, 0], metric_gt[mask]
                ).item() * len(batch)

            metric_depth_error += L1(metric_re[:, 0], metric_gt).item() * len(batch)
            sample_count += len(batch)

    return metric_depth_error / sample_count, metric_depth_error_under3 / sample_count


for name, dataloader in test_loaders.items():
    avg_l1, u3_l1 = evaluate(dataloader)

    if name in EXPERIMENT.train:
        print(f"[train] {name.name}")
    else:
        print(f"{name.name}")
    print(f"| L1     :", avg_l1)
    print(f"| L1 <3m :", u3_l1)

    print()
