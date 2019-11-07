import os
import re
from pathlib import Path
from typing import List, Tuple, Callable, Dict
from torch.utils.data import Dataset

from . import LanguageDataset

def load_texts(path:str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.readlines()
    except Exception as e:
        raise "Can't open the file"




class WangDataset():

    def __init__(self, sentence_pairs: Dict[str, List[str]]): 
        pass
        self.sentence_pairs = sentence_pairs

    @classmethod
    def from_text(cls, path_to_text_file: str):
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
        print("Number of lines in the text file: {}".format(len(lines)))
        for line in lines:
            search_obj = re.search(r"answer:\s\[[\'\"](.+)[\'\"]\]\svariable:\s(.+)", line)
            if search_obj == None:
                print("Can't parse the given text -- `{}`".format(line))
                continue
            else:
                if search_obj.group(1) != None:
                    th = search_obj.group(1) 
                    sentence_pairs["th"].append(th)
                if search_obj.group(2) != None:
                    en = search_obj.group(2)
                    sentence_pairs["en"].append(en)
           
        return cls(sentence_pairs)

    def get_language_pair_datasets(self, src_lang:str, tgt_lang:str):

        src_dataset = LanguageDataset(self.sentence_pairs[src_lang]),
        tgt_dataset = LanguageDataset(self.sentence_pairs[tgt_lang])

        return src_dataset, tgt_dataset