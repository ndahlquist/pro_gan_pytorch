
import os
import shutil
import urllib.request

import Dataloader
import tqdm


def maybe_download(transforms=None):
    torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
    directory = os.path.expanduser(torch_home + "/.datasets/pexels_landscapes/")
    if not os.path.exists(directory):
        print("Downloading dataset.")
        os.makedirs(directory)
        try:
            file_list = urllib.request.urlopen("https://storage.googleapis.com/pexels-landscapes/list.txt").read().split()

            for filename in tqdm.tqdm(file_list):
                filename = filename.decode("utf-8")
                with open(directory + filename, 'wb') as f:
                    url = "https://storage.googleapis.com/pexels-landscapes/" + filename
                    f.write(urllib.request.urlopen(url).read())
        except Exception as e:
            shutil.rmtree(directory)
            raise e
    return Dataloader.FlatDirectoryImageDataset(directory, transform=transforms)


if __name__ == "__main__":
    dataset = maybe_download()
    print(len(dataset))
