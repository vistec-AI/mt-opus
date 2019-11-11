import os
import re
from pathlib import Path
from typing import List, Tuple, Callable, Dict, Optional
from torch.utils.data import Dataset
from tqdm import tqdm
from . import LanguageDataset


from functools import partial
from pythainlp.tokenize import word_tokenize
_pythainlp_tokenize = partial(word_tokenize, engine="newmm", keep_whitespace=False)


def load_texts(path:str, number_of_lines:int = None):
    try:
        if number_of_lines == None:
            with open(path, "r", encoding="utf-8") as f: 
                return f.readlines()
        else:
            with open(path, "r", encoding="utf-8") as f: 
                return [ next(f) for x in range(number_of_lines) ]
    except Exception as e:
        raise "Can't open the file"




class WangDataset():

    def __init__(self, sentence_pairs: Dict[str, List[str]], sentence_pairs_lengths,
                src_tokenize=_pythainlp_tokenize,
                tgt_tokenize=_pythainlp_tokenize): 
        pass
        self.sentence_pairs = sentence_pairs
        self.sentence_pairs_lengths = sentence_pairs_lengths
        self.src_tokenize = src_tokenize
        self.tgt_tokenize = tgt_tokenize

    @classmethod
    def from_text(cls, path_to_text_file: str,
                src_lang,
                tgt_lang,
                src_tokenize=_pythainlp_tokenize,
                tgt_tokenize=_pythainlp_tokenize,
                number_of_lines:int = None):
        """
        Maps sentence pairs in the following format to a list of tuple of source and target sentence
        
        For example, 
            convert the line
                "answer: ['มันเป็นลูกแมวจริงๆ ในที่สุด'] variable: And it really was a kitten, after all."
                to a dictionary
                { "th": "มันเป็นลูกแมวจริงๆ ในที่สุด", "en": "And it really was a kitten, after all." }

        """
        
        lines = load_texts(path_to_text_file, number_of_lines)

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
                sent = dict()
                if search_obj.group(1) != None:
                    sent["th"] = search_obj.group(1).replace("?", "")

                if search_obj.group(2) != None:
                    sent["en"] =  search_obj.group(2)


                
                src_toks = src_tokenize(sent[src_lang])
                tgt_toks = tgt_tokenize(sent[tgt_lang])
                    
                sentence_pairs[src_lang].append(' '.join(src_toks))
                sentence_pairs_lengths[src_lang].append(len(src_toks))
                

                sentence_pairs[tgt_lang].append(' '.join(tgt_toks))
                sentence_pairs_lengths[tgt_lang].append(len(tgt_toks))



        return cls(sentence_pairs=sentence_pairs,
                   sentence_pairs_lengths=sentence_pairs_lengths,
                   src_tokenize=src_tokenize,
                   tgt_tokenize=tgt_tokenize)

    def get_language_pair_datasets(self, src_lang:str, tgt_lang:str, src_dict, tgt_dict):

        src_dataset = LanguageDataset(self.sentence_pairs[src_lang], src_dict)
        tgt_dataset = LanguageDataset(self.sentence_pairs[tgt_lang], tgt_dict)


        
        src_lengths = self.sentence_pairs_lengths[src_lang]
        tgt_lengths = self.sentence_pairs_lengths[tgt_lang]
        return src_dataset, src_lengths, tgt_dataset, tgt_lengths