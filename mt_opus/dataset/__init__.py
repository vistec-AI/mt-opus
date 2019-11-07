from typing import List
from torch.utils.data import Dataset, ConcatDataset


class LanguageDataset(ConcatDataset):
    """Store list of sentences"""

    def __init__(self, sentences: List[str]):
        # store a list of sentence List[str]
        self.sentences = sentences
    
    
    def __len__(self):
        return len(self.sentences)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.sentences[idx]

        return sample