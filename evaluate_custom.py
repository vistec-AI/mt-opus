#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""
import argparse
from functools import partial
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader

import fairseq

from fairseq.models.transformer import (
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS
) 

from fairseq import checkpoint_utils, data, options, tasks, bleu, options, progress_bar, utils
from fairseq.tasks import FairseqTask
from fairseq.tasks.translation import load_langpair_dataset
from fairseq.sequence_generator import SequenceGenerator
from fairseq.meters import StopwatchMeter, TimeMeter

from fairseq.data import FairseqDataset, LanguagePairDataset
import sentencepiece as spm

from pythainlp.tokenize import word_tokenize
from tqdm import tqdm_notebook, tqdm

from mt_opus.dataset.wang import WangDataset
from utils import str2bool


_pythainlp_tokenize = partial(word_tokenize, engine="newmm", keep_whitespace=False)
# intit spm
bpe_model_opensubtitles = spm.SentencePieceProcessor()
bpe_model_opensubtitles.Load('./data/sentencepiece_models/spm.opensubtitles.v2.model')


N_BEST = 5

def evaluate(model_path,
             src_dict_path,
             tgt_dict_path,
             tgt_dict_path_newmm,
             src_lang,
             tgt_lang,
             src_tok_type,
             tgt_tok_type,
             beam_size,
             remove_bpe,
             dataset,
             use_cuda,
             use_tokenizer,
             n_examples,
             n_best=N_BEST):

    

    state = checkpoint_utils.load_checkpoint_to_cpu(model_path)
    args = vars(state['args'])
    args['data'] = '/root/mt-opus/' + args['data']
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task([model_path], arg_overrides=args, task=None)
    
    src_dict = data.Dictionary()
    src_dict.add_from_file(src_dict_path)

    tgt_dict = data.Dictionary()
    tgt_dict.add_from_file(tgt_dict_path)

    tgt_dict_newmm = data.Dictionary()
    tgt_dict_newmm.add_from_file(tgt_dict_path_newmm)







    # Target always `newmm`
    scorer = bleu.Scorer(tgt_dict_newmm.pad(), tgt_dict_newmm.eos(), tgt_dict_newmm.unk())

    
#     print('len(src_dict) = ', len(src_dict))
#     print('len(tgt_dict) = ', len(tgt_dict))
    
    device = torch.device("cuda" if use_cuda else "cpu")

    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=beam_size
        )
        model.to(device)
    
    _generator = SequenceGenerator(tgt_dict=tgt_dict, beam_size=beam_size)
#     print('Inference from {} to {}'.format(src_lang, tgt_lang))
    
    if n_examples == None:
        n_examples = len(examples[src_lang])

        
    list_translation_results = []

    #TODO: Change from 1 example to mini-batch of size N

    for i, src_text in tqdm_notebook(enumerate(examples[src_lang][:n_examples]), total=n_examples):

        tgt_text = examples[tgt_lang][i]
        
#         print(i+1)
#         print('Source text :', src_text)
        if use_tokenizer:
            if src_tok_type == 'newmm':
                src_toks = _pythainlp_tokenize(src_text)
            elif src_tok_type == 'sentencepiece':
                src_toks = bpe_model_opensubtitles.EncodeAsPieces(src_text)
                
            tgt_toks = _pythainlp_tokenize(tgt_text)
            
        else:
            src_toks = src_text.split(' ')
            tgt_toks = tgt_text.split(' ')

#         print('src_toks :', src_toks)
        
#         print('tgt_toks :', tgt_toks)

        src_indices = [src_dict.index(t) for t in src_toks] + [src_dict.eos()]
        src_len = [len(src_indices)]
        
        tgt_indices = [tgt_dict_newmm.index(t) for t in tgt_toks] + [tgt_dict_newmm.eos()]
        tgt_len = [len(tgt_indices)]
        
  
        src_indices_tensor = torch.LongTensor(src_indices).to(device).view(1, -1)
        src_len_tensor = torch.LongTensor(src_len).to(device).view(1, -1)
        tgt_indices_tensor = torch.IntTensor(tgt_indices).to(device).view(1, -1)
        tgt_len_tensor = torch.IntTensor(tgt_len).to(device).view(1, -1)
     
                
#         print('src indices:', src_indices_tensor.size(), src_indices_tensor)
#         print('src len    :', src_len_tensor.size(), src_len_tensor)
#         print('tgt indices:', tgt_indices_tensor.size(), tgt_indices_tensor)
#         print('tgt len    :', tgt_len_tensor.size(), tgt_len_tensor)

        sample = {
            'net_input': {
                    'src_tokens': src_indices_tensor, 'src_lengths': src_len_tensor
            },
        }
#         print("sample", sample)
        prefix_tokens = None
        hypos =  _generator.generate(models=models, sample=sample, prefix_tokens=prefix_tokens)
        
        
        num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
    
    
        target_tokens = tgt_indices_tensor
        
#         print("num_generated_tokens", num_generated_tokens)
        for j, hypo in enumerate(hypos[0][:n_best]):
            
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_text,
                        alignment=None,
                        align_dict=None,
                        tgt_dict=tgt_dict,
                        remove_bpe=remove_bpe,
                    )
            
#             print("tgt str", tgt_text)

#             print("hypo_str (before)", hypo_str)
#             print("hypo_tokens (before):", hypo_tokens)
            if remove_bpe is not None:
                hypo_toks = _pythainlp_tokenize(hypo_str)
#                 print("hypo_toks (after)", hypo_toks)
                hypo_tokens = [tgt_dict_newmm.index(t) for t in hypo_toks] + [tgt_dict_newmm.eos()]
                hypo_tokens_tensor = torch.IntTensor(hypo_tokens)
#                 print("hypo_tokens (new)", hypo_tokens)
            else: 
                hypo_tokens_tensor = hypo_tokens
#             print("hypo_str", hypo_str)
            if j == 0: # select first result
                
                list_translation_results.append(
                    { "src_text": src_text, 
                      "hypo_text": hypo_str,
                      "tgt_text": tgt_text
                    })

                scorer.add(target_tokens, hypo_tokens_tensor)
#             print("\n")
#         print('\n----\n')
    print("\n\nBLEU Score:")
    print(scorer.result_string())
    print("\n\n\n")
    return scorer, list_translation_results

DATASET_NAMES = ["wang"]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_name", type=str, default="wang", help="Name of test dataset (e.g `wang`)")
    parser.add_argument("--examples_path", type=str, help="Path to the file storing dataset withno language id suffix (e.g `data/wang/wang.sent`)")
    parser.add_argument("--n_examples", type=int, default=None)

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--src_lang", type=str)
    parser.add_argument("--tgt_lang", type=str)
    parser.add_argument("--src_dict_path", type=str)
    parser.add_argument("--tgt_dict_path", type=str)
    parser.add_argument("--tgt_dict_path_newmm", type=str)

    parser.add_argument("--remove_bpe", type=str, default=None, help="If target token type is SentencePiece, specify this argument as `sentencepiece` if not specigy is as`None`")
    parser.add_argument("--use_tokenizer", type=str2bool, help="Either to use the tokenizer (newmm, sentencepiece) to pretokenize source, target sentences before feed to the NMT model")
    parser.add_argument("--use_cuda", type=str2bool, help="Either to use GPU or not", default=False)
    parser.add_argument("--tgt_tok_type", type=str, help="Either newmm, or sentencepiece_opensubtitles")
    parser.add_argument("--src_tok_type", type=str, help="Either newmm, sentencepiece_opensubtitles")

    args = parser.parse_args()
    examples = {"th": [], "en": []}



    if args.dataset_name == "wang":

        dataset = WangDataset.from_text(path_to_text_file=args.examples_path)
       
        src_dataset, src_lengths, tgt_dataset, tgt_lengths = dataset.get_language_pair_datasets(args.src_lang, args.tgt_lang)

    
        src_dict = data.Dictionary()
        src_dict.add_from_file(args.src_dict_path)

        tgt_dict = data.Dictionary()
        tgt_dict.add_from_file(args.tgt_dict_path)

        tgt_dict_newmm = data.Dictionary()
        tgt_dict_newmm.add_from_file(args.tgt_dict_path_newmm)

        print("Create LanguagePairDataset.")

        print()
        lang_pair_dataset = LanguagePairDataset(
                                src=src_dataset,
                                src_sizes=src_lengths,
                                tgt=tgt_dataset,
                                src_dict=src_dict,
                                tgt_dict=tgt_dict,
                                left_pad_source=False,
                                left_pad_target=False,
                                max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS,
                                max_target_positions=DEFAULT_MAX_TARGET_POSITIONS,)
        
        
        print("Done.")
        print(lang_pair_dataset[0])
        print(lang_pair_dataset[0]["source"][0])

        exit()
    else:
        print("Argument dataset_name is invalid (please only specify a name in this list: {})".format(DATASET_NAMES))
        exit()

    # model_name = Path(model_path).stem
    print(args)
    print('\nStart evaluation.')
    scorer, list_translation_results = evaluate(
                                        dataset=lang_pair_dataset,
                                        n_examples=None,
                                        use_cuda=args.use_cuda,
                                        model_path=args.model_path,
                                        src_dict=src_dict,
                                        tgt_dict=tgt_dict,
                                        tgt_dict_newmm=tgt_dict_newmm,
                                        src_lang=args.src_lang,
                                        tgt_lang=args.tgt_lang,
                                        use_tokenizer=args.use_tokenizer,
                                        beam_size=args.beam_size,
                                        remove_bpe=args.remove_bpe,
                                        src_tok_type=args.src_tok_type,
                                        tgt_tok_type=args.tgt_tok_type)

    print('\Done evaluation.')
