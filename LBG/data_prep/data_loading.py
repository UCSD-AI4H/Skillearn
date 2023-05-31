from __future__ import annotations

import platform
from typing import TYPE_CHECKING, NamedTuple, Optional, Dict, List, Optional, Union

from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from ethicml.data import create_genfaces_dataset
from ethicml.vision.data import LdColorizer

from .adult import load_adult_data
from .celeba import CelebA, CelebAttrs
from .dataset_wrappers import LdAugmentedDataset
from .misc import shrink_dataset, train_test_split
from .perturbed_adult import load_perturbed_adult
from .ssrp import SSRP
from .transforms import NoisyDequantize, Quantize

import argparse
import torch
from tap import Tap
from typing_extensions import Literal



# if TYPE_CHECKING:
#     from nifr.configs import SharedArgs



    
class SharedArgs(Tap):
    # General data set settings

    dataset: Literal["adult", "cmnist", "celeba", "ssrp", "genfaces"] = "celeba"
    device: Union[str, torch.device] = "cuda"

    data_pcnt: float = 1.0  # data pcnt should be a real value > 0, and up to 1
    task_mixing_factor: float = 0.0  # How much of meta train should be mixed into task train?
    pretrain: bool = True  # Whether to perform unsupervised pre-training.
    pretrain_pcnt: float = 0.4
    test_pcnt: float = 0.2

    # Adult data set feature settings
    drop_native: bool = True
    drop_discrete: bool = False

    # Colored MNIST settings
    scale: float = 0.02
    greyscale: bool = False
    background: bool = False
    black: bool = True
    binarize: bool = True
    rotate_data: bool = False
    shift_data: bool = False
    padding: int = 2  # by how many pixels to pad the input images
    quant_level: Literal["3", "5", "8"] = "8"  # number of bits that encode color
    input_noise: bool = True  # add uniform noise to the input

    # CelebA settings
    celeba_sens_attr: List[CelebAttrs] = ["Male"]
    celeba_target_attr: CelebAttrs = "Smiling"

    # # GenFaces settings
    # genfaces_sens_attr: GenfacesAttributes = "gender"
    # genfaces_target_attr: GenfacesAttributes = "emotion"

    # Optimization settings
    early_stopping: int = 30
    batch_size: int = 128
    test_batch_size: Optional[int] = 128
    num_workers: int = 0
    weight_decay: float = 0
    seed: int = 42
    data_split_seed: int = 888
    warmup_steps: int = 0
    gamma: float = 1.0  # Gamma value for Exponential Learning Rate scheduler.
    train_on_recon: bool = False  # whether to train the discriminator on recons or encodings
    recon_detach: bool = False  # Whether to apply the stop gradient operator to the reconstruction.

    # Evaluation settings
    eval_epochs: int = 40
    eval_lr: float = 1e-3
    encode_batch_size: int = 1000

    # Misc
    gpu: int = 0  # which GPU to use (if available)
    resume: Optional[str] = None
    evaluate: bool = False
    super_val: bool = False  # Train classifier on encodings as part of validation step.
    super_val_freq: int = 0  # how often to do super val, if 0, do it together with the normal val
    val_freq: int = 1000
    log_freq: int = 50
    root: str = ""
    use_wandb: bool = True
    results_csv: str = ""  # name of CSV file to save results to
    feat_attr: bool = False
    all_attrs: bool = False
    
    
    
    
    
    
    
    
    
    
    
     # General data set settings
    greyscale: bool = True

    # Optimization settings
    epochs: int = 50
    test_batch_size: int = 1000
    # lr: float = 1e-3
    lr: float=0.025
    data: str='../data'
    warmup: int=10
    dataset: str='celeba'
    # lr: float=0.025
    lr_min: float=0.0
    momentum: float=0.9
    wd: float=0.0003
    report_freq: int=100
    gpu: float=0
    # epochs: int=50
    init_ch: int=16
    cutout: bool=False
    cutout_len: int=16
    save: str='EXP'
    drop_path_prob: float=0.3
    train_portion: float=0.5
    cat_lr: float=0.0006
    cat_wd: float=0.001
    cat_steps: int=4
    unroll_steps: int=1
    lam: float=1
    num_experts: int=2
    gamma: float=1
    k: int=1
    load_balance: bool=False
    image_size: int=64

    def process_args(self):
        if not 0 < self.data_pcnt <= 1:
            raise ValueError("data_pcnt has to be between 0 and 1")
        if self.super_val_freq < 0:
            raise ValueError("frequency cannot be negative")

    def add_arguments(self):
        self.add_argument("-d", "--device", type=lambda x: torch.device(x), default="cpu")


class BaselineArgs(SharedArgs):
    
#     dataset: Literal["adult", "cmnist", "celeba", "ssrp", "genfaces"] = "celeba"
#     device: Union[str, torch.device] = "cuda"

#     data_pcnt: float = 1.0  # data pcnt should be a real value > 0, and up to 1
#     task_mixing_factor: float = 0.0  # How much of meta train should be mixed into task train?
#     pretrain: bool = False  # Whether to perform unsupervised pre-training.
#     pretrain_pcnt: float = 0.4
#     test_pcnt: float = 0.2

#     # Adult data set feature settings
#     drop_native: bool = True
#     drop_discrete: bool = False

#     # Colored MNIST settings
#     scale: float = 0.02
#     greyscale: bool = False
#     background: bool = False
#     black: bool = True
#     binarize: bool = True
#     rotate_data: bool = False
#     shift_data: bool = False
#     padding: int = 2  # by how many pixels to pad the input images
#     quant_level: Literal["3", "5", "8"] = "8"  # number of bits that encode color
#     input_noise: bool = True  # add uniform noise to the input

#     # CelebA settings
#     celeba_sens_attr: List[CelebAttrs] = ["Male"]
#     celeba_target_attr: CelebAttrs = "Smiling"

#     # # GenFaces settings
#     # genfaces_sens_attr: GenfacesAttributes = "gender"
#     # genfaces_target_attr: GenfacesAttributes = "emotion"

#     # Optimization settings
#     early_stopping: int = 30
#     batch_size: int = 128
#     test_batch_size: Optional[int] = 128
#     num_workers: int = 0
#     weight_decay: float = 0
#     seed: int = 42
#     data_split_seed: int = 888
#     warmup_steps: int = 0
#     gamma: float = 1.0  # Gamma value for Exponential Learning Rate scheduler.
#     train_on_recon: bool = False  # whether to train the discriminator on recons or encodings
#     recon_detach: bool = False  # Whether to apply the stop gradient operator to the reconstruction.

#     # Evaluation settings
#     eval_epochs: int = 40
#     eval_lr: float = 1e-3
#     encode_batch_size: int = 1000

#     # Misc
#     gpu: int = 0  # which GPU to use (if available)
#     resume: Optional[str] = None
#     evaluate: bool = False
#     super_val: bool = False  # Train classifier on encodings as part of validation step.
#     super_val_freq: int = 0  # how often to do super val, if 0, do it together with the normal val
#     val_freq: int = 1000
#     log_freq: int = 50
#     root: str = ""
#     use_wandb: bool = True
#     results_csv: str = ""  # name of CSV file to save results to
#     feat_attr: bool = False
#     all_attrs: bool = False
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#     # General data set settings
#     greyscale: bool = True

#     # Optimization settings
#     epochs: int = 50
#     test_batch_size: int = 1000
#     # lr: float = 1e-3
#     lr: float=0.025
#     data: str='../data'
#     warmup: int=10
#     dataset: str='celeba'
#     # lr: float=0.025
#     lr_min: float=0.0
#     momentum: float=0.9
#     wd: float=0.0003
#     report_freq: int=100
#     gpu: float=0
#     # epochs: int=50
#     init_ch: int=16
#     cutout: bool=False
#     cutout_len: int=16
#     save: str='EXP'
#     drop_path_prob: float=0.3
#     train_portion: float=0.5
#     cat_lr: float=0.0006
#     cat_wd: float=0.001
#     cat_steps: int=4
#     unroll_steps: int=1
#     lam: float=1
#     num_experts: int=2
#     gamma: float=1
#     k: int=1
#     load_balance: bool=False
#     image_size: int=64
    
    
    
    
    # Misc settings
    # method: BASELINE_METHODS = "naive"
    pred_s: bool = False
    save_dir: str = "experiments/baseline"
    
    
    
    def process_args(self):
        # if not 0 < self.data_pcnt <= 1:
        #     raise ValueError("data_pcnt has to be between 0 and 1")
        # if self.super_val_freq < 0:
        #     raise ValueError("frequency cannot be negative")
    # def process_args(self):
        self.method = "naive"
        if self.method == "kamiran":
            if self.dataset == "cmnist":
                raise ValueError(
                    "Kamiran & Calders reweighting scheme can only be used with binary sensitive and target attributes."
                )
            elif self.task_mixing_factor % 1 == 0:
                raise ValueError(
                    "Kamiran & Calders reweighting scheme can only be used when there is at least one sample available for each sensitive/target attribute combination."
                )

        return super().process_args()

    # def add_arguments(self):
    #     self.add_argument("-d", "--device", type=lambda x: torch.device(x), default="cpu")
    


    
    
__all__ = ["DatasetTriplet", "load_dataset"]


class DatasetTriplet(NamedTuple):
    pretrain: Optional[Dataset]
    task: Dataset
    task_train: Dataset
    s_dim: int
    y_dim: int


def load_dataset(args: SharedArgs) -> DatasetTriplet:
    assert args.pretrain
    pretrain_data: Dataset
    test_data: Dataset
    train_data: Dataset
    data_root = args.root or find_data_dir()
    # print(data_root)
    # data_root = "~/LBG/data"

    # =============== get whole dataset ===================
    if args.dataset == "cmnist":
        base_aug = [transforms.ToTensor()]
        data_aug = []
        if args.rotate_data:
            data_aug.append(transforms.RandomAffine(degrees=15))
        if args.shift_data:
            data_aug.append(transforms.RandomAffine(degrees=0, translate=(0.11, 0.11)))
        if args.padding > 0:
            base_aug.insert(0, transforms.Pad(args.padding))
        if args.quant_level != "8":
            base_aug.append(Quantize(int(args.quant_level)))
        if args.input_noise:
            base_aug.append(NoisyDequantize(int(args.quant_level)))
        train_data = MNIST(root=data_root, download=True, train=True)

        pretrain_len = round(args.pretrain_pcnt * len(train_data))
        train_len = len(train_data) - pretrain_len
        pretrain_data, train_data = random_split(train_data, lengths=(pretrain_len, train_len))

        test_data = MNIST(root=data_root, download=True, train=False)

        colorizer = LdColorizer(
            scale=args.scale,
            background=args.background,
            black=args.black,
            binarize=args.binarize,
            greyscale=args.greyscale,
        )

        pretrain_data = LdAugmentedDataset(
            pretrain_data,
            ld_augmentations=colorizer,
            num_classes=10,
            li_augmentation=True,
            base_augmentations=data_aug + base_aug,
        )
        train_data = LdAugmentedDataset(
            train_data,
            ld_augmentations=colorizer,
            num_classes=10,
            li_augmentation=False,
            base_augmentations=data_aug + base_aug,
        )
        test_data = LdAugmentedDataset(
            test_data,
            ld_augmentations=colorizer,
            num_classes=10,
            li_augmentation=True,
            base_augmentations=base_aug,
        )

        args.y_dim = 10
        args.s_dim = 10

    elif args.dataset == "ssrp":
        image_size = 64
        transform = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
        if args.quant_level != "8":
            transform.append(Quantize(int(args.quant_level)))
        if args.input_noise:
            transform.append(NoisyDequantize(int(args.quant_level)))
        transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform)

        pretrain_data = SSRP(data_root, pretrain=True, download=True, transform=transform)
        train_test_data = SSRP(data_root, pretrain=False, download=True, transform=transform)

        train_data, test_data = train_test_split(train_test_data, train_pcnt=(1 - args.test_pcnt))

        args.y_dim = train_test_data.num_classes
        args.s_dim = pretrain_data.num_classes

    elif args.dataset == "celeba":

        image_size = 64
        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
        # if args.quant_level != "8":
        #     transform.append(Quantize(int(args.quant_level)))
        # if args.input_noise:
        #     transform.append(NoisyDequantize(int(args.quant_level)))
        transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform)

        unbiased_pcnt = args.test_pcnt + args.pretrain_pcnt
        unbiased_data = CelebA(
            root=data_root,
            sens_attrs=args.celeba_sens_attr,
            target_attr_name=args.celeba_target_attr,
            biased=False,
            mixing_factor=args.task_mixing_factor,
            unbiased_pcnt=unbiased_pcnt,
            download=False,
            transform=transform,
            seed=args.data_split_seed,
        )

        pretrain_len = round(args.pretrain_pcnt / unbiased_pcnt * len(unbiased_data))
        test_len = len(unbiased_data) - pretrain_len
        pretrain_data, test_data = random_split(unbiased_data, lengths=(pretrain_len, test_len))

        train_data = CelebA(
            root=data_root,
            sens_attrs=args.celeba_sens_attr,
            target_attr_name=args.celeba_target_attr,
            biased=True,
            mixing_factor=args.task_mixing_factor,
            unbiased_pcnt=unbiased_pcnt,
            download=False,
            transform=transform,
            seed=args.data_split_seed,
        )

        args.y_dim = 1
        args.s_dim = unbiased_data.s_dim

    elif args.dataset == "genfaces":

        image_size = 64
        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
        if args.quant_level != "8":
            transform.append(Quantize(int(args.quant_level)))
        if args.input_noise:
            transform.append(NoisyDequantize(int(args.quant_level)))
        transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform)

        unbiased_pcnt = args.test_pcnt + args.pretrain_pcnt
        unbiased_data = create_genfaces_dataset(
            root=data_root,
            sens_attr_name=args.genfaces_sens_attr,
            target_attr_name=args.genfaces_target_attr,
            biased=False,
            mixing_factor=args.task_mixing_factor,
            unbiased_pcnt=unbiased_pcnt,
            download=True,
            transform=transform,
            seed=args.data_split_seed,
        )

        pretrain_len = round(args.pretrain_pcnt / unbiased_pcnt * len(unbiased_data))
        test_len = len(unbiased_data) - pretrain_len
        pretrain_data, test_data = random_split(unbiased_data, lengths=(pretrain_len, test_len))

        train_data = create_genfaces_dataset(
            root=data_root,
            sens_attr_name=args.genfaces_sens_attr,
            target_attr_name=args.genfaces_target_attr,
            biased=True,
            mixing_factor=args.task_mixing_factor,
            unbiased_pcnt=unbiased_pcnt,
            download=True,
            transform=transform,
            seed=args.data_split_seed,
        )

        args.y_dim = 1
        args.s_dim = unbiased_data.s_dim

    elif args.dataset == "adult":
        if args.input_noise:
            pretrain_data, test_data, train_data = load_perturbed_adult(args)
        else:
            pretrain_data, test_data, train_data = load_adult_data(args)

        args.y_dim = 1
        args.s_dim = 1
    else:
        raise ValueError("Invalid choice of dataset.")

    if 0 < args.data_pcnt < 1:
        pretrain_data = shrink_dataset(pretrain_data, args.data_pcnt)
        train_data = shrink_dataset(train_data, args.data_pcnt)
        test_data = shrink_dataset(test_data, args.data_pcnt)

    return DatasetTriplet(
        pretrain=pretrain_data,
        task=test_data,
        task_train=train_data,
        s_dim=args.s_dim,
        y_dim=args.y_dim,
    )


def find_data_dir() -> str:
    """Find data directory for the current machine based on predefined mappings."""
    data_dirs = {
        "fear": "/mnt/data0/data",
        "hydra": "/mnt/archive/shared/data",
        "m900382.inf.susx.ac.uk": "/Users/tk324/PycharmProjects/NoSINN/data",
        "turing": "/srv/galene0/shared/data",
    }
    name_of_machine = platform.node()  # name of machine as reported by operating system
    return data_dirs.get(name_of_machine, "data")
