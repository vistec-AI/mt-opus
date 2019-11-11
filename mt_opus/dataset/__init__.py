from typing import List

import torch
from torch.utils.data import Dataset, ConcatDataset


class LanguageDataset(Dataset):
    """Store list of sentences"""

    def __init__(self, sentences: List[str], dictionary):
        # store a list of sentence List[str]
        self.sentences = sentences
        self.sentences_tensor = []
        for sentence in sentences:

            tokens = sentence.split(" ")

            sentence_tensor = [ dictionary.index(t) for t in tokens ]
            sentence_tensor.extend([dictionary.eos()])
            sentence_tensor = torch.LongTensor(sentence_tensor)
            self.sentences_tensor.append(sentence_tensor)

    def __len__(self):
        return len(self.sentences_tensor)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.sentences_tensor[idx]

        return sample