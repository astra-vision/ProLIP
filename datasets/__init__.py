from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .imagenetv2 import ImageNetV2
from .imagenet_sketch import ImageNetSketch
from .imagenet_r import ImageNetR
from .imagenet_a import ImageNetA


dataset_list = {
                "oxford_pets": OxfordPets,
                "eurosat": EuroSAT,
                "ucf101": UCF101,
                "sun397": SUN397,
                "caltech101": Caltech101,
                "dtd": DescribableTextures,
                "fgvc": FGVCAircraft,
                "food101": Food101,
                "oxford_flowers": OxfordFlowers,
                "stanford_cars": StanfordCars,
                "imagenetv2": ImageNetV2,
                "imagenet_sketch": ImageNetSketch,
                "imagenet_rendition": ImageNetR,
                "imagenet_adversarial": ImageNetA,
                }


def build_dataset(dataset, subsample, root_path):
    return dataset_list[dataset](subsample,root_path)