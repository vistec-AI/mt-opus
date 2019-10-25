import argparse
import os
from pathlib import Path
from functools import partial
from pythainlp.tokenize import word_tokenize

_pythainlp_tokenize = partial(word_tokenize, engine="newmm", keep_whitespaces=False)


def load_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Path to the directory stored sentence paris (e,g, `./data/wang`)")
    parser.add_argument("--input_prefix", type=str, help="The name of text file where language ID is exclude (e.g. `wang_2018` )")

    parser.add_argument("--src_lang", type=str, default="th")
    parser.add_argument("--tgt_lang", type=str, default="en")
    parser.add_argument("--src_type", type=str, default="newmm")
    parser.add_argument("--tgt_type", type=str, default="newmm")

    args = parser.parse_args()
    
    input_dir = args.input_dir
    input_prefix = args.input_prefix
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    src_type = args.src_type
    tgt_type = args.tgt_type

    src, tgt = (src_lang, src_type), (tgt_lang, tgt_type)

    print("Load file")

    src_path = os.path.join(input_dir, input_prefix + ".{}.txt".format(src_lang))    
    tgt_path = os.path.join(input_dir, input_prefix + ".{}.txt".format(tgt_lang))

    sentences = {
        '{}'.format(src_lang): load_text_file(src_path),
        '{}'.format(tgt_lang): load_text_file(tgt_path)
    }

    
    output_dir = os.path.join(input_dir, "tokenized", "{}-{}".format(src_lang, tgt_lang), "{}-{}".format(src_type, tgt_type))
    os.mkdir(output_dir)
    
    output_path = os.path.join(output_dir, "tokenized_sentences.{}".format(src_lang))
    print("\nWrite file of tokenized sentences for the source language ({})\nto {}".format(src_lang, output_path))
    with open(output_path, "w", encoding="utf-8") as f:
        _tokenize = _pythainlp_tokenize if src_type == "newmm" else None
        for sentence in sentences[lang][:-1]:
            tokens = _tokenize(sentence)
            f.write(" ".join(tokens + "\n")
        tokens = _tokenize(sentences[-1])
        f.write(" ".join(tokens)
    print("\nDone.")

    output_path = os.path.join(output_dir, "tokenized_sentences.{}".format(tgt_lang))
    print("\nWrite file of tokenized sentences for the target language ({})\nto {}".format(tgt_lang, output_path))
    with open(output_path, "w", encoding="utf-8") as f:
        _tokenize = _pythainlp_tokenize if tgt_type == "newmm" else None
        for sentence in sentences[lang][:-1]:
            tokens = _tokenize(sentence)
            f.write(" ".join(tokens + "\n")
        tokens = _tokenize(sentences[-1])
        f.write(" ".join(tokens)
    print("\nDone.")
