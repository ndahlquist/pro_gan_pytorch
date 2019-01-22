import os
import random
import shutil
import urllib.request

import cv2
import torch
from torch.utils.data import Dataset

_torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
_directory = os.path.expanduser(_torch_home + "/datasets/a_thousand_li/")

_url = "https://github.com/ndahlquist/pro_gan_pytorch/raw/a_thousand_li/data/a_thousand_li_of_rivers_and_mountains.jpg"


class AThousandLiDataset(Dataset):

    def __init__(self, shape=128):
        self.output_shape = 16

        self.full_image = cv2.imread(_directory + 'a_thousand_li.jpg')[:, 2000:-1000, :]

    def __len__(self):
        return 5000

    def __getitem__(self, i):

        crop_size = int(self.output_shape * random.uniform(2.0, 4.0))

        x = random.randint(0, self.full_image.shape[1] - crop_size)
        y = random.randint(0, self.full_image.shape[0] - crop_size)

        crop = self.full_image[y:y+crop_size, x:x+crop_size, ...]

        crop = cv2.resize(crop, (self.output_shape, self.output_shape), interpolation=cv2.INTER_AREA)

        # Random horizontal flip.
        if random.randint(0, 1) == 0:
            crop = cv2.flip(crop, 1)

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        return torch.tensor(crop).float().permute(2, 0, 1) / 256



def maybe_download():
    if not os.path.exists(_directory):
        print("Downloading dataset.")
        os.makedirs(_directory)
        try:
            with open(_directory + 'a_thousand_li.jpg', 'wb') as f:
                f.write(urllib.request.urlopen(_url).read())

        except Exception as e:
            shutil.rmtree(_directory)
            raise e
    return AThousandLiDataset()


if __name__ == "__main__":
    dataset = maybe_download()
    for i in range(30):
        crop = dataset[0].permute(1,2,0).numpy()
        print(crop.shape)
        cv2.imshow('im', crop)
        cv2.waitKey()
