import os
import re
from pathlib import Path
from typing import List, Tuple, Callable, Dict
from torch.utils.data import Dataset
from tqdm import tqdm
from . import LanguageDataset


from functools import partial
from pythainlp.tokenize import word_tokenize
_pythainlp_tokenize = partial(word_tokenize, engine="newmm", keep_whitespace=False)


def load_texts(path:str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.readlines()
    except Exception as e:
        raise "Can't open the file"




class WangDataset():

    def __init__(self, sentence_pairs: Dict[str, List[str]], sentence_pairs_lengths, tokenize=_pythainlp_tokenize): 
        pass
        self.sentence_pairs = sentence_pairs
        self.sentence_pairs_lengths = sentence_pairs_lengths
        self.tokenize = tokenize
    @classmethod
    def from_text(cls, path_to_text_file: str, tokenize=_pythainlp_tokenize):
        """
        Maps sentence pairs in the following format to a list of tuple of source and target sentence
        
        For example, 
            convert the line
                "answer: ['มันเป็นลูกแมวจริงๆ ในที่สุด'] variable: And it really was a kitten, after all."
                to a dictionary
                { "th": "มันเป็นลูกแมวจริงๆ ในที่สุด", "en": "And it really was a kitten, after all." }

        """
        
        lines = load_texts(path_to_text_file)

        sentence_pairs = {
            "th": [],
            "en": []
        }
        sentence_pairs_lengths = {
            "th": [],
            "en": []
        }

        print("Number of lines in the text file: {}".format(len(lines)))
        for line in tqdm(lines):
            search_obj = re.search(r"answer:\s\[[\'\"](.+)[\'\"]\]\svariable:\s(.+)", line)
            if search_obj == None:
                print("Can't parse the given text -- `{}`".format(line))
                continue
            else:
                if search_obj.group(1) != None:
                    th = search_obj.group(1) 
                    toks = tokenize(th)
                    sentence_pairs["th"].append(' '.join(toks))
                    sentence_pairs["th"].append(len(toks))
                if search_obj.group(2) != None:
                    en = search_obj.group(2)
                    toks = tokenize(en)
                    sentence_pairs["en"].append(' '.join(toks))
                    sentence_pairs["en"].append(len(toks))


        return cls(sentence_pairs=sentence_pairs, sentence_pairs_lengths=sentence_pairs_lengths, tokenize=tokenize)

    def get_language_pair_datasets(self, src_lang:str, tgt_lang:str):

        src_dataset = LanguageDataset(self.sentence_pairs[src_lang])
        tgt_dataset = LanguageDataset(self.sentence_pairs[tgt_lang])


        
        src_lengths = self.sentence_pairs_lengths[src_lang]
        tgt_lengths = self.sentence_pairs_lengths[tgt_lang]
        return src_dataset, src_lengths, tgt_dataset, tgt_lengths