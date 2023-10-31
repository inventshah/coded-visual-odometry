# Coded Visual Odometry

### Basics

Generate coded images with `coded-generator.py`. Make sure to edit the camera parameters on line 171 based on your PSFs.

Add new experiments to `configopts.py`.

Run the trainer: `python trainer.py -c <your experiment name>`.

Evaluate a model: `python tester.py -c <your experiment name> -s <path to checkpoint>`

Generate metric depth maps: `python generator.py -c <your experiment name> -s <path to checkpoint>`

Commands also have a `-d` or `--DATASET` flag to set the root of the data folders. Results will be put into each datasets folder next to their coded image.

### Datasets

Download the [ICL-NUIM dataset](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html). Unzip each into a folder inside a dataset root. For example,

```
datasets/
	office_traj2/
		depth/
		rgb/
	office_traj3/
		...
	living_traj0/
		...
```

Edit `DatasetName` in `configopts.py` to change what datasets are available. Currently, those datasets must follow the ICL format; however, any dataset with metric depth maps available work. In `coded-generator.py` call `Camera.process_folder` with various parameters depending on the folder structure of units of the dataset depth. In `data.py`, make a new `torch.utils.data.Dataset` class corresponding to the structure. Then, use that class when loading datasets in the trainer/tester/generator files.
