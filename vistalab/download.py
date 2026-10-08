import csv
import io
import urllib.request
import zipfile
from pathlib import Path

import kagglehub
from datasets import load_dataset
from huggingface_hub import hf_hub_download

CACHE = Path(__file__).parent.parent / ".cache"


def hf_rows(repo, config, split):
    return load_dataset(repo, config, split=split).to_list()


def hf_sample(repo, config, split, n):
    rows = load_dataset(repo, config, split=split).shuffle(seed=13)
    return rows.select(range(min(n, len(rows)))).to_list()


def download(url):
    path = CACHE / url.split("/")[-1]
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def hf_file(repo, name):
    return Path(hf_hub_download(repo, name, repo_type="dataset"))


def kaggle_file(dataset, name):
    return Path(kagglehub.dataset_download(dataset)) / name


def read_zip(path, name):
    with zipfile.ZipFile(path) as z:
        member = next(n for n in z.namelist() if n.endswith(name))
        return z.read(member).decode("utf-8")


def read_tsv(text):
    return list(csv.DictReader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE))
