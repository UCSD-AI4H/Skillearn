import os
from pathlib import Path
from typing import List, Literal, Sequence

import pandas as pd
import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_file_from_google_drive
from torchvision.transforms import ToTensor

from ethicml.preprocessing import (
    ProportionalSplit,
    SequentialSplit,
    get_biased_subset,
    query_dt,
    train_test_split,
)
from ethicml.preprocessing.domain_adaptation import make_valid_variable_name
from ethicml.utility import DataTuple


CelebAttrs = Literal[
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]


class CelebA(VisionDataset):
    """Large-scale CelebFaces Attributes (CelebA) Dataset

    <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>
    Adapted from torchvision.datasets to enable the loading of data triplets
    while removing superfluous (for our purposes) elements of the dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
            Defaults to ``attr``. If empty, ``None`` will be returned as target.
        sens_attr (string): Attribute to use as the sensitive attribute.
        target_attr (string): Attribute to use as the target attribute.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "celeba"

    file_list = [
        (
            "0B7EVK8r0v71pZjFTYXZWM3FlRnM",  # File ID
            "00d2c5bc6d35e252742224ab0c1e8fcb",  # MD5 Hash
            "img_align_celeba.zip",  # Filename
        ),
        (
            "0B7EVK8r0v71pblRyaVFSWGxPY0U",
            "75e246fa4810816ffd6ee81facbd244c",
            "list_attr_celeba.txt",
        ),
        (
            "0B7EVK8r0v71pY0NSMzRuSXJEVkk",
            "d32c9cbf5e040fd4025c592c306e6668",
            "list_eval_partition.txt",
        ),
    ]

    disc_feature_groups = {
        "beard": ["5_o_Clock_Shadow", "Goatee", "Mustache", "No_Beard"],
        "hair_color": ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"],
        # not yet sorted into categories:
        "Arched_Eyebrows": ["Arched_Eyebrows"],
        "Attractive": ["Attractive"],
        "Bags_Under_Eyes": ["Bags_Under_Eyes"],
        "Bangs": ["Bangs"],
        "Big_Lips": ["Big_Lips"],
        "Big_Nose": ["Big_Nose"],
        "Blurry": ["Blurry"],
        "Bushy_Eyebrows": ["Bushy_Eyebrows"],
        "Chubby": ["Chubby"],
        "Double_Chin": ["Double_Chin"],
        "Eyeglasses": ["Eyeglasses"],
        "Heavy_Makeup": ["Heavy_Makeup"],
        "High_Cheekbones": ["High_Cheekbones"],
        "Male": ["Male"],
        "Mouth_Slightly_Open": ["Mouth_Slightly_Open"],
        "Narrow_Eyes": ["Narrow_Eyes"],
        "Oval_Face": ["Oval_Face"],
        "Pale_Skin": ["Pale_Skin"],
        "Pointy_Nose": ["Pointy_Nose"],
        "Receding_Hairline": ["Receding_Hairline"],
        "Rosy_Cheeks": ["Rosy_Cheeks"],
        "Sideburns": ["Sideburns"],
        "Smiling": ["Smiling"],
        "hair_type": ["Bald", "Straight_Hair", "Wavy_Hair"],
        "Wearing_Earrings": ["Wearing_Earrings"],
        "Wearing_Hat": ["Wearing_Hat"],
        "Wearing_Lipstick": ["Wearing_Lipstick"],
        "Wearing_Necklace": ["Wearing_Necklace"],
        "Wearing_Necktie": ["Wearing_Necktie"],
        "Young": ["Young"],
    }

    def __init__(
        self,
        root: str,
        biased: bool,
        mixing_factor: float,
        unbiased_pcnt: float,
        sens_attrs: List[CelebAttrs],
        target_attr_name: CelebAttrs,
        transform=None,
        target_transform=None,
        download: bool = False,
        seed: int = 42,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        base = Path(self.root) / self.base_folder
        partition_file = base / "list_eval_partition.txt"
        # partition: information about which samples belong to train, val or test
        partition = pd.read_csv(
            partition_file, delim_whitespace=True, header=None, index_col=0, names=["partition"]
        )
        # attrs: all attributes with filenames as index
        attrs = pd.read_csv(base / "list_attr_celeba.txt", delim_whitespace=True, header=1)
        all_data = pd.concat([partition, attrs], axis="columns", sort=False)
        # the filenames are used for indexing; here we turn them into a regular column
        all_data = all_data.reset_index(drop=False).rename(columns={"index": "filenames"})

        attr_names = list(attrs.columns)

        if len(sens_attrs) == 2 and ("Male" in sens_attrs) and ("Young" in sens_attrs):
            self.s_dim = 4
            gender = (all_data["Male"] + 1) // 2  # map from {-1, 1} to {0, 1}
            age = (all_data["Young"] + 1) // 2  # map from {-1, 1} to {0, 1}
            sens_attr = (gender * 2 + age).to_frame(name="agender")
            print(
                "\n".join(
                    f"agender: {gender * 2 + age}, gender: {gender}, age: {age}"
                    for age in (0, 1)
                    for gender in (0, 1)
                )
            )
        elif len(sens_attrs) > 1:
            if any(sens_attr_name not in attr_names for sens_attr_name in sens_attrs):
                raise ValueError(f"at least one of {sens_attrs} does not exist as an attribute.")
            # only use those samples where exactly one of the specified attributes is true
            all_data = all_data.loc[((all_data[sens_attrs] + 1) // 2).sum(axis="columns") == 1]
            self.s_dim = len(sens_attrs)
            # perform the reverse operation of one-hot encoding
            data_only_sens = all_data[sens_attrs]
            data_only_sens.columns = list(range(self.s_dim))
            sens_attr = data_only_sens.idxmax(axis="columns").to_frame(name=",".join(sens_attrs))
        else:
            sens_attr_name = sens_attrs[0]
            if sens_attr_name not in attr_names:
                raise ValueError(f"{sens_attr_name} does not exist as an attribute.")
            sens_attr = all_data[[sens_attr_name]]
            sens_attr = (sens_attr + 1) // 2  # map from {-1, 1} to {0, 1}
            self.s_dim = 1

        if target_attr_name not in attr_names:
            raise ValueError(f"{target_attr_name} does not exist as an attribute.")
        target_attr = all_data[[target_attr_name]]
        target_attr = (target_attr + 1) // 2  # map from {-1, 1} to {0, 1}

        s_and_y = [target_attr_name] + sens_attrs
        filename = all_data.drop(s_and_y, axis="columns")

        all_dt = DataTuple(x=filename, s=sens_attr, y=target_attr)

        if seed == 888:
            # NOTE: the sequential split does not shuffle
            unbiased_dt, biased_dt, _ = SequentialSplit(train_percentage=unbiased_pcnt)(all_dt)
        else:
            unbiased_dt, biased_dt = train_test_split(all_dt, unbiased_pcnt, random_seed=seed)

        if biased:
            if self.s_dim > 1:
                if mixing_factor not in (0, 1):
                    raise ValueError("multi-valued s can't be used with mixing")
                if mixing_factor == 0:
                    s_for_y0 = (1, 2)
                    s_for_y1 = (0, 3)
                else:
                    s_for_y0 = (0, 3)
                    s_for_y1 = (1, 2)
                biased_dt = select_biased_subset(biased_dt, s_for_y0=s_for_y0, s_for_y1=s_for_y1)
            else:
                biased_dt, _ = get_biased_subset(
                    data=biased_dt, mixing_factor=mixing_factor, unbiased_pcnt=0, seed=seed
                )
            filename, sens_attr, target_attr = biased_dt
        else:
            filename, sens_attr, target_attr = unbiased_dt

        self.filename = filename["filenames"].to_numpy()
        self.other_attrs = filename.drop("filenames", axis=1)
        self.sens_attr = torch.as_tensor(sens_attr.to_numpy())
        self.target_attr = torch.as_tensor(target_attr.to_numpy())

    def _check_integrity(self):
        base = Path(self.root) / self.base_folder
        for (_, md5, filename) in self.file_list:
            fpath = base / filename
            ext = fpath.suffix
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(str(fpath), md5):
                return False

        ## Should check a hash of the images
        # return (base / "img_align_celeba").is_dir()
        return (base / "img_align_celeba").is_dir()

    def download(self):
        import zipfile

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(
                file_id, os.path.join(self.root, self.base_folder), filename, md5
            )

        with zipfile.ZipFile(
            os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r"
        ) as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    def __getitem__(self, index):
        X = Image.open(
            os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index])
        )
        S = self.sens_attr[index]
        target = self.target_attr[index]

        if self.transform is not None:
            X = self.transform(X)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return X, S, target

    def __len__(self):
        return len(self.sens_attr)

    def extra_repr(self):
        lines = ["Target type: {target_type}", "Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)


def select_biased_subset(
    data: DataTuple, s_for_y0: Sequence[int], s_for_y1: Sequence[int]
) -> DataTuple:
    """Get the subset where s and y are equal to the given pairs."""
    s_name = data.s.columns[0]
    y_name = data.y.columns[0]

    s_name = make_valid_variable_name(s_name)
    y_name = make_valid_variable_name(y_name)
    query_str = f"({y_name} == 0 & (" + " | ".join(f"{s_name} == {s}" for s in s_for_y0) + "))"
    query_str += f" | ({y_name} == 1 & (" + " | ".join(f"{s_name} == {s}" for s in s_for_y1) + "))"
    print(query_str)
    biased = query_dt(data, query_str)
    # antibiased = query_dt(data, f"~({query_str})")
    return biased  # , antibiased


if __name__ == "__main__":
    from torch.utils.data import DataLoader, random_split

    train_data = CelebA(
        root=r"./data",
        biased=True,
        mixing_factor=1.0,
        unbiased_pcnt=0.4,
        download=False,
        transform=ToTensor(),
    )

    split_size = round(0.4 * len(train_data))
    train_data, _ = random_split(train_data, lengths=(split_size, len(train_data) - split_size))
    assert len(train_data) == split_size
    train_loader = DataLoader(train_data, batch_size=9)
    x, s, y = next(iter(train_loader))
    
    assert x.size(0) == 9
    assert x.size(0) == s.size(0) == y.size(0)
