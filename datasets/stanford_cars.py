import os
from scipy.io import loadmat

from .oxford_pets import OxfordPets
from .utils import Datum, DatasetBase


template = ['a photo of a {}.']


class StanfordCars(DatasetBase):

    dataset_dir = 'stanford_cars'

    def __init__(self, subsample, root):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_StanfordCars.json')

        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)
        train, val, test = OxfordPets.subsample_classes(train,
                                                        val,
                                                        test,
                                                        subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)
    
    def read_data(self, image_dir, anno_file, meta_file):
        anno_file = loadmat(anno_file)['annotations'][0]
        meta_file = loadmat(meta_file)['class_names'][0]
        items = []

        for i in range(len(anno_file)):
            imname = anno_file[i]['fname'][0]
            impath = os.path.join(self.dataset_dir, image_dir, imname)
            label = anno_file[i]['class'][0, 0]
            label = int(label) - 1 # convert to 0-based index
            classname = meta_file[label][0]
            names = classname.split(' ')
            year = names.pop(-1)
            names.insert(0, year)
            classname = ' '.join(names)
            item = Datum(
                impath=impath,
                label=label,
                classname=classname
            )
            items.append(item)
        
        return items

    def generate_fewshot_dataset_(self,num_shots, split):

        print('num_shots is ',num_shots)
        if split == "train":
            few_shot_data = self.generate_fewshot_dataset(self.train_x, num_shots=num_shots)
        elif split == "val":
            few_shot_data = self.generate_fewshot_dataset(self.val, num_shots=num_shots)

        return few_shot_data