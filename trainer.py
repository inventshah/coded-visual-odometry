import argparse
import os
import sys
import time


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import wandb


from configopts import get_experiment, DatasetName
from data import ICLDataset
from unets import init_weights, count_parameters


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


parser = argparse.ArgumentParser()
parser.add_argument("--CONFIG", "-c", type=str, required=True)
parser.add_argument("--SAVE_EVERY", "-s", type=int, default=25, required=False)
parser.add_argument(
    "--DATASET",
    "-d",
    type=str,
    default="datasets",
    required=False,
)
parser.add_argument(
    "--SAVEDIR",
    "-s",
    type=str,
    default="checkpoints",
    required=False,
)


args = parser.parse_args(sys.argv[1:])
DATASET_DIR = args.DATASET

EXPERIMENT = get_experiment(args.CONFIG)


experiment_name = EXPERIMENT.__class__.__name__.split("/")[-1]
CHECKPOINT_DIR = os.path.join(args.SAVEDIR, experiment_name)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# datasets

icl_datasets = {
    name: ICLDataset(DATASET_DIR, name.name, EXPERIMENT.coded, cache=True)
    for name in DatasetName
}

train_dataset: ConcatDataset = ConcatDataset(
    [icl_datasets[name] for name in EXPERIMENT.train]
)
train_loader = DataLoader(
    train_dataset, batch_size=EXPERIMENT.batch_size, shuffle=True, num_workers=4
)

test_loaders = {
    name: DataLoader(dataset, batch_size=EXPERIMENT.batch_size, shuffle=True)
    for name, dataset in icl_datasets.items()
}

# model

model = EXPERIMENT.model().to(device)
init_weights(model)

model_optimizer = torch.optim.Adam(model.parameters(), lr=EXPERIMENT.learning_rate)

L1 = nn.L1Loss()

wandb.init(project="cvod2", name=experiment_name, config={**vars(args)})
print(repr(EXPERIMENT))
print(f"Training model with {count_parameters(model)}")


def wandbimg(img: torch.Tensor, vmax=6.5):
    out = np.clip(img.detach().cpu().numpy(), 0, vmax) / vmax * 255
    return wandb.Image(out.astype(np.uint8))


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
    model.train()

    ground_truth_depth_map = wandbimg(metric_gt[0])
    reconstructed_depth_map = wandbimg(metric_re[0, 0])

    return (
        metric_depth_error / sample_count,
        metric_depth_error_under3 / sample_count,
        ground_truth_depth_map,
        reconstructed_depth_map,
    )


NUM_TEST_SETS = len(DatasetName) - len(EXPERIMENT.train)
previous_test_error = 9999999.0
for i in range(EXPERIMENT.epochs):
    start = time.monotonic()

    total_error = 0
    for batch in train_loader:
        model_optimizer.zero_grad()

        reconstruction = EXPERIMENT.post_forward(model(batch["Coded"].to(device)))
        metric_gt = batch["Depth"].to(device)

        ground_truth = EXPERIMENT.depth.metric_to_output(metric_gt)

        error = EXPERIMENT.compute_loss(ground_truth, reconstruction, i)

        error.backward()
        model_optimizer.step()
        total_error += error.item()

    test_artifacts = {}
    total_avg_l1 = 0
    total_u3_l1 = 0
    for name, dataloader in test_loaders.items():
        avg_l1, u3_l1, gt_depth_map, re_depth_map = evaluate(dataloader)

        test_artifacts[f"{name.name}: L1"] = avg_l1
        test_artifacts[f"{name.name}: L1 <3m"] = u3_l1
        if i % 5 == 0:
            test_artifacts[f"{name.name}: ground truth"] = gt_depth_map
            test_artifacts[f"{name.name}: reconstructed"] = re_depth_map

        if name not in EXPERIMENT.train:
            total_avg_l1 += avg_l1
            total_u3_l1 += u3_l1

    iterate_values = {
        "train_error": total_error / len(train_loader),
        "test_L1": total_avg_l1 / NUM_TEST_SETS,
        "test_L1_under3": total_u3_l1 / NUM_TEST_SETS,
        **test_artifacts,
    }

    if total_u3_l1 / NUM_TEST_SETS < previous_test_error:
        previous_test_error = total_u3_l1 / NUM_TEST_SETS
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best.pt")

    wandb.log(iterate_values)

    if i % args.SAVE_EVERY == 0:
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/recon_{i}.pt")

    end = time.monotonic()
    print(f"Epoch={i}: loss={error.item()} :: {end - start:.3f}s")


torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/recon_end.pt")

wandb.finish()
