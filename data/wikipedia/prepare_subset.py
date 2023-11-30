# saves the openwebtext dataset to a binary file for training. following was helpful:
# taken from Karpathy's nanogpt project https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py

# python 3.11 required, 3.12 incompatible with aiohttp
# the following need to be installed with homebrew: apache-arrow, libxml2, libxslt, rust
# the following need to be installed with pip: tqdm, tiktoken, pyarrow, aiohttp, apache_beam, datasets, mwparserfromhell

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
#num_proc = 1

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
#num_proc_load_dataset = num_proc

if __name__ == '__main__':
    print("loading data from hf")
    dataset = load_dataset("wikipedia", "20220301.en", #num_proc=num_proc_load_dataset,
                           split='train', streaming=True)
    dataset_subset = dataset.take(100) # first few examples

    print("data loaded")
    # wikipedia by default only contains the 'train' split, so create a test split
    split_dataset_subset = dataset_subset["train"].train_test_split(test_size=0.05, seed=12345, shuffle=True)
    split_dataset_subset['val'] = split_dataset_subset.pop('test') # rename the test split to val
    #n = len(dataset_subset)
    #train_data = dataset_subset[:int(n*0.9)]
    #val_data = dataset_subset[int(n*0.9):]

    #print("getting encodings")
    #enc = tiktoken.get_encoding("gpt2")
    #print("train encodings")
    #train_ids = enc.encode_ordinary(train_data)
    #print("val encodings")
    #val_ids = enc.encode_ordinary(val_data)
    #print(f"train has {len(train_ids):,} tokens")
    #print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    #print("exporting")
    #train_ids = np.array(train_ids, dtype=np.uint16)
    #val_ids = np.array(val_ids, dtype=np.uint16)
    #train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    #val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
