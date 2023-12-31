from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Type

import torch
from torch import nn

from unets import U_Net, AttU_Net, U_NetSmall

L1 = nn.L1Loss()
L2 = nn.MSELoss()


def weighted_mse_loss(input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor):
    return torch.sum(weight * (input - target) ** 2) / torch.sum(weight)


DepthStyle = Enum("DepthStyle", ["metric", "phi", "normalized"])
DatasetName = Enum(
    "DatasetName", ["office_traj2", "office_traj3", "living_traj1", "living_traj0"]
)

# helpful metric units
cm = 1e-2
mm = 1e-3


class DepthSpace(ABC):
    """convert metric depth maps to and from model output maps"""

    @abstractmethod
    def output_to_metric(self, out: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def metric_to_output(self, met: torch.Tensor) -> torch.Tensor:
        ...

    def __repr__(self):
        return self.__class__.__name__


class MetricDepth(DepthSpace):
    """Basic metric depth representation"""

    def output_to_metric(self, out: torch.Tensor):
        return out

    def metric_to_output(self, met: torch.Tensor):
        return met


class NormalizedDepth(DepthSpace):
    """
    Represent depth as 0-1 based on a maximum distance.
    """

    def __init__(self, max_value: float = 7):
        self.max_value = max_value

    def output_to_metric(self, out: torch.Tensor):
        return out * self.max_value

    def metric_to_output(self, met: torch.Tensor):
        return met / self.max_value

    def __repr__(self):
        return f"{self.__class__.__name__}(max_value={self.max_value})"


class DiopterDepth(DepthSpace):
    """
    Represent depth as defocus terms away from the focal plane.

    Based on PhaseCam3D equation 5 and normalized by `max_diopter`
    """

    def __init__(
        self,
        f_number: float = 17,
        focal_length: float = 50 * mm,
        focus_distance: float = 85 * cm,
        max_diopter: int = 15,
    ):
        self.f_number = f_number
        self.focal_length = focal_length

        self.R = focal_length / (2 * f_number)
        self.focus_distance = focus_distance
        self.k = 2 * torch.pi / 530e-9
        self.max_diopter = max_diopter

    def output_to_metric(self, out: torch.Tensor):
        out2 = out * self.max_diopter * 2 - self.max_diopter
        Wm = out2 / self.k
        depth = 1 / (1 / self.focus_distance + 2 * Wm / self.R**2)
        return depth

    def metric_to_output(self, met: torch.Tensor):
        depth = torch.clamp(met, 1 * mm, None)  # prevent 0 depth
        inv = 1 / depth
        sub = inv - 1 / self.focus_distance
        div = sub / 2
        Wm = div * self.R**2
        Phi2 = Wm * self.k
        return (Phi2 + self.max_diopter) / (2 * self.max_diopter)

    def __repr__(self):
        return f"\
{self.__class__.__name__}( \
f_number={self.f_number}, \
focal_length={self.focal_length*100:.1f}cm, \
focus_distance={self.focus_distance*100:.1f}cm, \
max_diopter={self.max_diopter} \
)"


def get_experiment(class_name: str):
    return [
        cls  # type: ignore
        for cls in Experiment.__subclasses__()
        if cls.__name__ == class_name and cls != Experiment
    ][0]()


class Experiment(ABC):
    name: str
    # params
    model: Type[nn.Module]
    depth: DepthSpace
    # train
    epochs: int
    batch_size: int
    learning_rate: float

    # datasets
    train: List[DatasetName]
    coded: str

    @abstractmethod
    def compute_loss(
        self, ground_truth: torch.Tensor, reconstruction: torch.Tensor, idx: int = 0
    ):
        ...

    @abstractmethod
    def post_forward(self, reconstruction: torch.Tensor):
        ...

    def __repr__(self):
        out = f"{self.__class__.__name__}(\n"
        out += f"\tmodel={self.model.__name__}\n"
        out += f"\tdepth={self.depth!r}\n"
        out += f"\tLR={self.learning_rate}\n"
        out += f"\tepochs={self.epochs}\n"
        out += f"\tbatch-size={self.batch_size}\n"
        out += f"\ttrain-set={[item.name for item in self.train]}\n"
        out += f"\tcoded={self.coded!r}\n"
        out += ")"
        return out


class SimpleDiopter(Experiment):
    model = U_Net
    depth = DiopterDepth(max_diopter=13)
    epochs = 200
    batch_size = 8
    learning_rate = 1e-4
    train = [DatasetName.office_traj2]
    coded = "Codedphasecam-27Linear"

    def post_forward(self, reconstruction: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(reconstruction)

    def compute_loss(
        self, ground_truth: torch.Tensor, reconstruction: torch.Tensor, idx: int = 0
    ) -> torch.Tensor:
        return L2(reconstruction[:, 0], ground_truth)


class SimpleNormalized(Experiment):
    model = U_Net
    depth = NormalizedDepth()
    epochs = 200
    batch_size = 8
    learning_rate = 1e-4
    train = [DatasetName.office_traj2]
    coded = "Codedphasecam-27Linear"

    def post_forward(self, reconstruction: torch.Tensor):
        return torch.sigmoid(reconstruction)

    def compute_loss(
        self, ground_truth: torch.Tensor, reconstruction: torch.Tensor, idx: int = 0
    ):
        return L2(reconstruction[:, 0], ground_truth)


class MetricWeightedLoss(Experiment):
    model = U_Net
    depth = MetricDepth()
    epochs = 200
    batch_size = 8
    learning_rate = 1e-4
    train = [DatasetName.office_traj2]
    coded = "Codedphasecam-27Linear"

    def post_forward(self, reconstruction: torch.Tensor):
        return reconstruction

    def compute_loss(
        self, ground_truth: torch.Tensor, reconstruction: torch.Tensor, idx: int = 0
    ):
        reconstruction_metric = self.depth.output_to_metric(reconstruction)
        ground_truth_metric = self.depth.output_to_metric(ground_truth)

        return weighted_mse_loss(
            reconstruction_metric[:, 0],
            ground_truth_metric,
            2 ** (-0.3 * ground_truth_metric),
        )


class DiopterWithMetricUnder3(Experiment):
    model = U_Net
    depth = DiopterDepth(max_diopter=13)
    epochs = 200
    batch_size = 8
    learning_rate = 1e-4
    train = [DatasetName.office_traj2]
    coded = "Codedphasecam-27Linear"

    def post_forward(self, reconstruction: torch.Tensor):
        return torch.sigmoid(reconstruction)

    def compute_loss(
        self, ground_truth: torch.Tensor, reconstruction: torch.Tensor, idx: int = 0
    ):
        if idx < 3:
            return L2(reconstruction[:, 0], ground_truth)

        reconstruction_metric = self.depth.output_to_metric(reconstruction)
        ground_truth_metric = self.depth.output_to_metric(ground_truth)

        mask = (ground_truth_metric > 0.5) & (ground_truth_metric < 3)
        return (
            L2(reconstruction[:, 0], ground_truth)
            + L1(reconstruction_metric[:, 0][mask], ground_truth_metric[mask]) / 6
        )


class DirectDiopterWithMetricUnder3(Experiment):
    model = U_Net
    depth = DiopterDepth(max_diopter=12)
    epochs = 200
    batch_size = 8
    learning_rate = 1e-4
    train = [DatasetName.office_traj2]
    coded = "Codedphasecam-27Linear"

    def post_forward(self, reconstruction: torch.Tensor):
        return reconstruction

    def compute_loss(
        self, ground_truth: torch.Tensor, reconstruction: torch.Tensor, idx: int = 0
    ):
        if idx < 5:
            return L2(reconstruction[:, 0], ground_truth)

        reconstruction_metric = self.depth.output_to_metric(reconstruction)
        ground_truth_metric = self.depth.output_to_metric(ground_truth)

        mask = (ground_truth_metric > 0.5) & (ground_truth_metric < 3)
        return (
            L2(reconstruction[:, 0], ground_truth)
            + L1(reconstruction_metric[:, 0][mask], ground_truth_metric[mask]) / 6
        )


class DirectNormalizedWithMetricUnder3(Experiment):
    model = U_Net
    depth = NormalizedDepth()
    epochs = 200
    batch_size = 8
    learning_rate = 5e-5
    train = [DatasetName.office_traj2]
    coded = "Codedphasecam-27Linear"

    def post_forward(self, reconstruction: torch.Tensor):
        return reconstruction

    def compute_loss(
        self, ground_truth: torch.Tensor, reconstruction: torch.Tensor, idx: int = 0
    ):
        if idx < 2:
            return L2(reconstruction[:, 0], ground_truth)

        reconstruction_metric = self.depth.output_to_metric(reconstruction)
        ground_truth_metric = self.depth.output_to_metric(ground_truth)

        mask = (ground_truth_metric > 0.5) & (ground_truth_metric < 3)
        return (
            L2(reconstruction[:, 0], ground_truth)
            + L1(reconstruction_metric[:, 0][mask], ground_truth_metric[mask]) / 6
        )
