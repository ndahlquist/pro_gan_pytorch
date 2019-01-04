
import os
import shutil
import urllib.request

import multiprocessing
import Dataloader
import tqdm

_torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
_directory = os.path.expanduser(_torch_home + "/.datasets/pexels_landscapes/")


def _download_file(filename):
    filename = filename.decode("utf-8")
    with open(_directory + filename, 'wb') as f:
        url = "https://storage.googleapis.com/pexels-landscapes/" + filename
        f.write(urllib.request.urlopen(url).read())


def maybe_download(transforms=None):
    if not os.path.exists(_directory):
        print("Downloading dataset.")
        os.makedirs(_directory)
        try:
            file_list = urllib.request.urlopen("https://storage.googleapis.com/pexels-landscapes/list.txt").read().split()

            # Download using multiple threads, and use tqdm to show progress.
            pool = multiprocessing.Pool()

            for _ in tqdm.tqdm(pool.imap_unordered(_download_file, file_list), total=len(file_list)):
                pass

        except Exception as e:
            shutil.rmtree(_directory)
            raise e
    return Dataloader.FlatDirectoryImageDataset(_directory, transform=transforms)


if __name__ == "__main__":
    dataset = maybe_download()
    print(len(dataset))
