import os

import cv2
import numpy as np
import torch
import tqdm

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_to_numpy_image(tensor: torch.Tensor):
    out = tensor.moveaxis(0, -1)
    return torch.clamp(out, 0, 255).to(torch.uint8).cpu().numpy()


class Camera:
    def __init__(
        self,
        name: str,
        psfs: torch.Tensor,
        depth_layers: torch.Tensor,
        use_nonlinear: bool = False,
    ):
        self.name = f"Coded{name}NonLinear" if use_nonlinear else f"Coded{name}Linear"
        n_depths, n_channels, height, width = psfs.shape

        if (
            n_channels != 3
            or n_depths != depth_layers.numel()
            or width != height
            or width % 2 == 0
        ):
            raise ValueError(f"PSF has wrong shape: {psfs.shape}")

        self.psfs = psfs.to(device)
        self.depth_layers = depth_layers.to(device)
        self.use_nonlinear = use_nonlinear

        self.padding = width // 2

    def capture(self, img: np.ndarray, metric_depth: np.ndarray) -> torch.Tensor:
        image = torch.from_numpy(img).moveaxis(-1, 0).to(torch.float32).to(device)
        depth = torch.from_numpy(metric_depth).to(device)

        if self.use_nonlinear:
            coded = self.nonlinear(image, depth)
        else:
            coded = self.linear(image, depth)
        return tensor_to_numpy_image(coded)

    def get_depth_layers(self, depth_map: torch.Tensor):
        quantized_depth = torch.bucketize(depth_map, self.depth_layers)
        return torch.stack(
            [quantized_depth == j for j in range(len(self.depth_layers))]
        )

    def linear(self, image: torch.Tensor, depth_map: torch.Tensor):
        """simple linear depth-dependent convolution"""
        depth_mask = self.get_depth_layers(depth_map)

        return torch.stack(
            [
                (
                    torch.sum(
                        torch.nn.functional.conv2d(
                            image[None, channel : channel + 1],
                            self.psfs[:, channel : channel + 1],
                            stride=1,
                            padding=self.padding,
                        )
                        * depth_mask,
                        dim=1,
                    )
                )
                for channel in range(3)
            ],
            dim=1,
        )[0]

    def single_psf_convolution(
        self, image: torch.Tensor, depth_idx: int, channel_idx: int
    ):
        return torch.nn.functional.conv2d(
            image,
            self.psfs[depth_idx : depth_idx + 1, channel_idx : channel_idx + 1],
            stride=1,
            padding=self.padding,
        )

    def nonlinear(self, img: torch.Tensor, depth_map: torch.Tensor, eps=1e-6):
        """perform non-linear blurring based on Ikoma et al. 2021 equation 5"""
        depth_mask = self.get_depth_layers(depth_map)
        K, _, _ = depth_mask.shape
        depth_mask = depth_mask.to(torch.float)

        depth_mask = torch.flip(depth_mask, dims=(0,))

        out = torch.zeros_like(img)

        img = img.to(torch.float) / 255.0

        depth_sum = torch.cumsum(depth_mask, dim=0)

        for channel in range(3):
            layered = img[channel : channel + 1] * depth_mask

            for k in range(K):
                E_k = self.single_psf_convolution(depth_sum[k][None, None], k, channel)

                l_k = self.single_psf_convolution(
                    layered[k][None, None], k, channel
                ) / (E_k + eps)
                for kp in range(k + 1, K):
                    E_kp = self.single_psf_convolution(
                        depth_sum[kp][None, None], kp, channel
                    )
                    a_kp = 1 - self.single_psf_convolution(
                        depth_mask[kp][None, None], kp, channel
                    ) / (E_kp + eps)
                    l_k = l_k * a_kp
                out[channel] = out[channel] + l_k

        return torch.clamp(out * 255, 0, 255)

    def process_folder(
        self,
        root: str,
        depth_folder: str = "depth",
        image_folder: str = "rgb",
        factor: int = 5000,
        subset: str = "",
    ):
        os.makedirs(os.path.join(root, self.name, subset), exist_ok=True)

        files = os.listdir(os.path.join(root, image_folder))
        for file in tqdm.tqdm(
            files,
            total=len(files),
            desc=root,
        ):
            image_file = os.path.join(root, image_folder, file)
            depth_file = os.path.join(root, depth_folder, file)

            image_bgr = cv2.imread(image_file)
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            raw_depth = cv2.imread(
                depth_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
            )
            if raw_depth is None:
                print(file, "is missing a depth file")
                continue
            metric_depth = raw_depth / factor

            coded_image_rgb = self.capture(image.astype(np.float32), metric_depth)
            coded_image_bgr = cv2.cvtColor(coded_image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(root, self.name, subset, file), coded_image_bgr)


def process_icl(cam: Camera, path: str):
    cam.process_folder(os.path.join(path, "office_traj3"))
    cam.process_folder(os.path.join(path, "office_traj2"))
    cam.process_folder(os.path.join(path, "living_traj0"))
    cam.process_folder(os.path.join(path, "living_traj1"))


# change the Camera parameters to match some pre-generated PSF file
# ensure the depth_layers matches to depths used to generate the PSFs
# the values should be in metric units, but if you edit Camera
# to quantize with a different space, you can change the units
camera = Camera(
    "phasecam-27",
    torch.from_numpy(np.moveaxis(np.load("phasecam-psfs-27.npy"), -1, 1)),
    torch.linspace(0.5, 6, 27),
    False,
)


process_icl(camera, "datasets")
