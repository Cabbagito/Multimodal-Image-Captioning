from numpy import array_split
from json import load
import random
import os
from tqdm import tqdm
from glob import glob
from imageio import imread
from concurrent.futures import ThreadPoolExecutor
from skimage.transform import resize


class Dataset:
    def __init__(self, data_dir, resize_shape, verbose):
        self.DATA_DIR = data_dir
        self.papers = []
        self.resize_shape = resize_shape
        self.verbose = verbose
        random.seed(42)

    def load(self, batch):
        data = []
        paper_batch = array_split(self.papers, self.num_batches)[batch]
        papers = array_split(paper_batch, self.num_workers)
        futures = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for idx, paper in enumerate(papers):
                futures.append(
                    executor.submit(
                        self.load_papers, paper, self.verbose if idx == 0 else False
                    )
                )
        for future in futures:
            data.extend(future.result())
        return data

    def load_papers(self, papers, verbose=False):
        data = []
        if verbose:
            papers = tqdm(papers, desc="Loading data")
        for paper in papers:
            try:
                with open(os.path.join(paper, "text.json"), "r") as f:
                    json = load(f)
                figures = glob(os.path.join(paper, "*.png"))
                for idx, figure in enumerate(figures):
                    figure = imread(figure)
                    data.append(
                        {
                            "id": json["id"],
                            "title": json["title"].strip().replace("\n", " "),
                            "abstract": json["abstract"].strip().replace("\n", " "),
                            "figure": resize(figure, self.resize_shape)
                            if self.resize_shape
                            else figure,
                            "label": json["text"][idx].strip().replace("\n", " "),
                        }
                    )
            except:
                pass
        return data

    def __getitem__(self, idx):
        return self.load(idx)

    def __len__(self):
        return self.num_batches


class CS10KDataset(Dataset):
    def __init__(
        self, data_dir, num_batches=100, resize_shape=None, verbose=True, num_workers=10
    ):
        self.num_batches = num_batches
        self.num_workers = num_workers
        super().__init__(data_dir, resize_shape, verbose)
        self.get_papers()
        random.shuffle(self.papers)

    def get_papers(self):
        for paper in os.listdir(self.DATA_DIR):
            self.papers.append(os.path.join(self.DATA_DIR, paper))


class MixDataset(Dataset):
    def __init__(
        self, data_dir, num_batches=100, resize_shape=None, verbose=True, num_workers=10
    ):
        self.num_batches = num_batches
        self.num_workers = num_workers
        super().__init__(data_dir, resize_shape, verbose)
        self.get_papers()
        random.shuffle(self.papers)

    def get_papers(self):
        for archive in os.listdir(self.DATA_DIR):
            for year in os.listdir(os.path.join(self.DATA_DIR, archive)):
                for paper in os.listdir(os.path.join(self.DATA_DIR, archive, year)):
                    self.papers.append(
                        os.path.join(self.DATA_DIR, archive, year, paper)
                    )
