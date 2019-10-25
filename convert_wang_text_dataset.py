import argparse
import json
import os
from pathlib import Path

from mt_opus.dataset.wang import WangDataset

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("--output_dir", type=str, default="./data/wang")


    args = parser.parse_args()
    output_dir = args.output_dir
    data_path = args.data_path
    data_name = Path(data_path).stem

    print("\nLoad wang dataset from: {path}\n".format(path=data_path))
    wang_dataset = WangDataset.from_text(args.data_path)
    sentence_pairs = wang_dataset.sentence_pairs
    print("\nNumber of sentence pairs: {}\n".format(len(sentence_pairs)))


    for lang in ["th", "en"]:
        data_path = os.path.join(output_dir, data_name + '.{}'.format(lang))
        print("\nWriting {} sentence to {}".format(lang, data_path))
        with open(data_path, "w", encoding="utf-8") as f:
            for pair in sentence_pairs[:-1]:
                f.write(pair[lang] + "\n")
            f.write(sentence_pairs[-1][lang])
    
        print("Done Writing sentences to file.")

