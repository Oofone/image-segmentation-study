from datasets.utils.mv_utils import Metadata, LabelConfig, build_metadata_map, build_label_metadata_map
from datasets.utils.mv_utils import enrich_label_metadata, nested_object_hook, build_label_config_from_path
from datasets.transforms import get_default_image_transform, get_default_mask_transform
from datasets.collator import BatchCollaterMappilaryVistasPTLightning
from utils.json_utils import read_json, write_json_objects
from utils.image_utils import open_image_as_PIL

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from pytorch_lightning import LightningDataModule

from typing import Optional, Callable, Tuple, List, Set

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn import Identity
from torch import Tensor

import copy
import os


# Constants
## Dataset specific

ALLOWED_VERSIONS = ["v1.2", "v2.0"]
SPLITS = {
    "LABELLED": {
        "splits": ["training", "validation"],
        "versions": ALLOWED_VERSIONS,
    },
    "UNLABELLED": {
        "splits": ["testing"],
        "versions": [],
    }
}

## Files

METADATA_FILE = "metadata.json"
IMAGES_DIR = "images"
LABELS_DIR = "labels"

# Dataset

class MapillaryVistas(Dataset):

    def _build_config_path(version: str = ALLOWED_VERSIONS[1]) -> str:
        return f"config_{version}.json"

    def _get_allowed_splits() -> Set[str]:
        allowed_splits = set()
        for val in SPLITS.values():
            allowed_splits = allowed_splits.union(set(val["splits"]))
        return allowed_splits

    def _get_labelled_splits() -> Set[str]:
        return set(SPLITS["LABELLED"]['splits'])

    def _get_unlabelled_splits() -> Set[str]:
        return set(SPLITS["UNLABELLED"]['splits'])

    def _build_metadata(root_path: str) -> List[Metadata]:
        print("Building Metadata")
        metadata = []

        print("Indexing labelled images:")
        for split in SPLITS["LABELLED"]["splits"]:
            print(f"Indexing images of split [{split}]")
            split_path = os.path.join(root_path, split)
            images_path = os.path.join(split_path, IMAGES_DIR)
            l_meta_map = build_metadata_map(images_path=images_path, split=split)
            for version in ALLOWED_VERSIONS:
                print(f"Indexing labels of split [{split}]; and version [{version}]")
                labels_path = os.path.join(split_path, version, LABELS_DIR)
                config_file = os.path.join(root_path, MapillaryVistas._build_config_path(version))
                label_metadata_map = build_label_metadata_map(labels_path=labels_path, label_config_path=config_file, version=version)
                l_meta_map_v = enrich_label_metadata(copy.deepcopy(l_meta_map), label_metadata_map)
                metadata.extend(list(l_meta_map_v.values()))

        print("Indexing unlabelled images:")
        for split in SPLITS["UNLABELLED"]["splits"]:
            print(f"Indexing split [{split}]")
            split_path = os.path.join(root_path, split)
            images_path = os.path.join(split_path, IMAGES_DIR)
            ul_meta_map = build_metadata_map(images_path=images_path, split=split)
            metadata.extend(list(ul_meta_map.values()))

        return metadata

    def _build_default_metadata_file_path(root_path: str) -> str:
        return os.path.join(root_path, METADATA_FILE)

    def __init__(
        self, split: str, root: str, metadata_path: str | None = None, version: str = "v2.0",
        image_transform: Callable[[Tensor], Tensor] = Identity(),
        target_transform: Callable[[Tensor], Tensor] = Identity(),
        resolution: Tuple[int] = (512, 512), save_metadata_on_build: bool = True,
    ) -> None:
        super().__init__()
        allowed_splits = MapillaryVistas._get_allowed_splits()
        assert split in allowed_splits, f"Split [{split}] is not in allowed splits: {allowed_splits}"
        assert (version in ALLOWED_VERSIONS) or (split in MapillaryVistas._get_unlabelled_splits()), f"Version must be in {ALLOWED_VERSIONS} for splits: {MapillaryVistas._get_labelled_splits()}; Got {version}"

        self.split = split
        self.root = root
        self.version = version
        self.resolution = resolution
        self.image_transform = image_transform
        self.target_transform = target_transform

        self.metadata = None
        if metadata_path is None:
            def_meta_file_path = MapillaryVistas._build_default_metadata_file_path(self.root)
            if os.path.isfile(def_meta_file_path):
                self.metadata = read_json(def_meta_file_path, nested_object_hook)
            else:
                self.metadata = MapillaryVistas._build_metadata(self.root)
                if save_metadata_on_build:
                    write_json_objects(self.metadata, def_meta_file_path)
        else:
            self.metadata: List[Metadata] = read_json(metadata_path, lambda x: Metadata(**x))

        self.metadata = list(filter(
            lambda meta: (meta.split == self.split) and ((not meta.has_label) or (meta.label_metadata.version == self.version)), self.metadata))
        print(f"Load Filtered {len(self.metadata)} data in {split}.")

    def __getitem__(self, index: int) -> List[Tensor]:
        meta = self.metadata[index]
        image = open_image_as_PIL(meta.file)
        image = self.image_transform(image)

        if meta.has_label:
            target = open_image_as_PIL(meta.label_metadata.label_file)
            target = self.target_transform(target)
            
            return {
                'name': meta.name,
                'image': image,
                'target': target,
                'target_labels': meta.label_metadata.labels_in_image,
                'target_labels_in_eval': meta.label_metadata.eval_labels_in_image,
            }

        return {
            'name': meta.name,
            'image': image,
            'target': None,
            'target_labels': None,
            'target_labels_in_eval': None,
        }

    def __len__(self):
        return len(self.metadata)


class MapillaryVistasDataModule(LightningDataModule):

    def __init__(
        self, metadata_path: Optional[str] = None, train_split: str='training', val_split: str='validation', test_split: str='testing',
        root: str = "data", version: str = "v2.0", resolution: Tuple[int] = (512, 512), image_transform: Optional[Callable[[Tensor], Tensor]] = None,
        target_transform: Optional[Callable[[Tensor], Tensor]] = None, batch_size: int = 8, val_batch_size: int=8, num_workers: int = 8,
        save_metadata_on_build: bool = True
    ) -> None:
        super().__init__()
        
        assert resolution is not None, f"Resolution parameter cannot be None; Got [{resolution}]"
        
        self.root = root
        self.version = version
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.metadata_path = metadata_path
        self.image_resolution = resolution
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.save_metadata_on_build = save_metadata_on_build

        self.image_transform = image_transform if image_transform is not None else get_default_image_transform(resolution)
        self.target_transform = target_transform if target_transform is not None else get_default_mask_transform(resolution)

        self.collator = BatchCollaterMappilaryVistasPTLightning()

    def setup(self, stage: Optional[str] = None) -> None:

        self.train_dataset = MapillaryVistas(
            root=self.root, split=self.train_split, metadata_path=self.metadata_path, version=self.version, image_transform=self.image_transform,
            target_transform=self.target_transform, save_metadata_on_build=self.save_metadata_on_build)

        self.val_dataset = MapillaryVistas(
            root=self.root, split=self.val_split, metadata_path=self.metadata_path, version=self.version, image_transform=self.image_transform,
            target_transform=self.target_transform, save_metadata_on_build=self.save_metadata_on_build)

        self.test_dataset = MapillaryVistas(
            root=self.root, split=self.test_split, metadata_path=self.metadata_path, version=self.version, image_transform=self.image_transform,
            target_transform=self.target_transform, save_metadata_on_build=self.save_metadata_on_build)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collator, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, num_workers=self.num_workers, collate_fn=self.collator, shuffle=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size, num_workers=self.num_workers, collate_fn=self.collator, shuffle=False)

    def get_CMAP(self) -> List[int]:
        config_path = os.path.join(self.root, MapillaryVistas._build_config_path(self.version))
        config: LabelConfig = build_label_config_from_path(config_path=config_path)

        CMAP = []
        for _, label in enumerate(config.labels):
            for color in label.color:
                CMAP.append(color)
        return CMAP


# The dictionary is used to map the feature type to the dataset type
# The key is a tuple of (visual_feature_type, audio_feature_type), ``None`` means using end-to-end encoder.
feature_type_to_dataset_type = {
    (None, None): MapillaryVistas
}
