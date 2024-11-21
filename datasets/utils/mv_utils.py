from utils.image_utils import get_image_resolution, open_image
from utils.json_utils import read_json

from typing import Optional, List, Tuple, Dict, Set
from dataclasses import dataclass

import numpy as np
import tqdm
import os


# Dataclasses

@dataclass
class Label:
    name: str
    readable: str
    instances: bool
    evaluate: bool
    color: List[int]


@dataclass
class LabelConfig:
    labels: List[Label]
    version: str
    mapping: str
    folder_structure: str


@dataclass
class LabelMetadata:
    version: str
    label_file: str
    labels_in_image: List[str]
    eval_labels_in_image: List[str]


@dataclass
class Metadata:
    name: str
    file: str
    split: str
    has_label: bool
    label_metadata: LabelMetadata | None
    original_resolution: Tuple[int]

# Utility functions

def build_label_config_from_path(config_path: str) -> LabelConfig:
    config_json = read_json(config_path)
    config_json["labels"] = [Label(**label) for label in config_json["labels"]]
    return LabelConfig(**config_json)


def get_label_arrays(label_config: LabelConfig) -> Tuple[np.array, Set[str]]:
    labels = []
    eval_labels = []
    for label in label_config.labels:
        labels.append(label.name)
        if label.evaluate:
            eval_labels.append(True)
        else:
            eval_labels.append(False)
    labels = np.array(labels)
    eval_labels_mask = np.array(eval_labels)
    assert labels.shape == eval_labels_mask.shape, "Error with labels and eval label masks"
    
    return (labels, set(labels[eval_labels_mask].tolist()))


def get_labels_in_image(label_path: str, label_arr: np.array) -> List[str]:
    input_lab_arr = open_image(label_path)
    return label_arr[np.unique(input_lab_arr)].tolist()


def build_metadata_map(images_path: str, split: str) -> Dict[str, Metadata]:
    image_paths = [os.path.join(images_path, im) for im in os.listdir(images_path)]

    return {
        im_path.split("/")[-1].split(".")[0] : Metadata(
            name = im_path.split("/")[-1].split(".")[0],
            file = im_path,
            split = split,
            has_label = False,
            label_metadata = None,
            original_resolution = get_image_resolution(im_path))
        for im_path in tqdm.tqdm(image_paths)
    }


def build_label_metadata(lab_path: str, version: str, lab_arr: np.array, eval_set: Set[str]) -> LabelMetadata:
    labels_im = get_labels_in_image(lab_path, lab_arr)
    labels_eval = [lab for lab in labels_im if lab in eval_set]
    return LabelMetadata(
        version = version,
        label_file = lab_path,
        labels_in_image = labels_im,
        eval_labels_in_image = labels_eval
    )


def build_label_metadata_map(labels_path: str, label_config_path: str, version: str) -> Dict[str, LabelMetadata]:
    label_paths = [os.path.join(labels_path, lab) for lab in os.listdir(labels_path)]
    label_config = build_label_config_from_path(label_config_path)
    lab_arr, eval_set = get_label_arrays(label_config=label_config)

    return {
        lab_path.split("/")[-1].split(".")[0] : build_label_metadata(
            lab_path=lab_path,
            version=version,
            lab_arr=lab_arr,
            eval_set=eval_set)
        for lab_path in tqdm.tqdm(label_paths)
    }


def enrich_label_metadata(metadata_map: Dict[str, Metadata], label_metadata_map: Dict[str, LabelMetadata]) -> Dict[str, Metadata]:
    for key, lab_meta in tqdm.tqdm(label_metadata_map.items()):
        if key in metadata_map:
            metadata_map[key].has_label = True
            metadata_map[key].label_metadata = lab_meta
    return metadata_map


def nested_object_hook(x: Dict):
    if type(x) == type(dict()):
        if "label_metadata" in x:
            x["label_metadata"] = nested_object_hook(x["label_metadata"])
            return Metadata(**x)
        elif "label_file" in x:
            return LabelMetadata(**x)
        else:
            return x
    else:
        return x
