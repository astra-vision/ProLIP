import os
from .utils import Datum, DatasetBase
from .oxford_pets import OxfordPets


template = ['a photo of a person doing {}.']


class UCF101(DatasetBase):

    dataset_dir = 'ucf101'

    def __init__(self, subsample, root):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'UCF-101-midframes')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_UCF101.json')

        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)

        train, val, test = OxfordPets.subsample_classes(train,
                                                    val,
                                                    test,
                                                    subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)
    
    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')[0] # trainlist: filename, label
                action, filename = line.split('/')
                label = cname2lab[action]

                elements = re.findall('[A-Z][^A-Z]*', action)
                renamed_action = '_'.join(elements)

                filename = filename.replace('.avi', '.jpg')
                impath = os.path.join(self.image_dir, renamed_action, filename)

                item = Datum(
                    impath=impath,
                    label=label,
                    classname=renamed_action
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