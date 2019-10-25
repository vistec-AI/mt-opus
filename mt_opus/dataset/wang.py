import os
import re
from pathlib import Path
from typing import List, Tuple, Callable, Dict

def load_texts(path:str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.readlines()
    except Exception as e:
        raise "Can't open the file"

class WangDataset(object):

    def __init__(self, sentence_pairs: List[Dict[str, str]]):
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

        sentence_pairs = []
        print("Number of lines in the text file: {}".format(len(lines)))
        for line in lines:
            search_obj = re.search(r"answer:\s\[[\'\"](.+)[\'\"]\]\svariable:\s(.+)", line)
            if search_obj == None:
                print("Can't parse the given text -- `{}`".format(line))
                continue
            lang_pair_dict = {
                "th": search_obj.group(1),
                "en": search_obj.group(2)
            }
            sentence_pairs.append(lang_pair_dict)
           
        return cls(sentence_pairs)