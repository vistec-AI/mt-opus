#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""
import argparse
import torch
import fairseq

from fairseq import checkpoint_utils, data, options, tasks, bleu, options, progress_bar, utils
from fairseq.sequence_generator import SequenceGenerator
from fairseq.meters import StopwatchMeter, TimeMeter

from utils import str2bool

"""
data-path    - path that stored (e.g. `./data/opensubtitles_bin/newmm-newmm/th-en`) 
               inside that path there are dictionary (`dict.en.th`, `dict.th.txt`)
--path(s)    - to model file(s), colon separated
--remove-bpe - remove BPE tokens before scoring (can be set to sentencepiece)
--quite      - only print final scores
--beam	     - beam size




"""
N_BEST = 5

def evaluate(model_path,
             src_dict_path,
             tgt_dict_path,
             src_tok_type,
             tgt_tok_type
             beam_size,
             remove_bpe):

    use_cuda = torch.cuda.is_available() and not args.cpu

    state = checkpoint_utils.load_checkpoint_to_cpu(model_path)
    args = vars(state['args'])
    args['data'] = '/root/mt-opus/' + args['data']
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task([model_path], arg_overrides=args, task=None)
    
    src_dict = data.Dictionary()
    src_dict.add_from_file(src_dict_path)

    tgt_dict = data.Dictionary()
    tgt_dict.add_from_file(tgt_dict_path)
    
    
    scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())

    print('len(src_dict) = ', len(src_dict))
    print('len(tgt_dict) = ', len(tgt_dict))
    
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=beam_size
        )
        if use_cuda:
            model.cuda()
    
    generator = SequenceGenerator(tgt_dict=tgt_dict, beam_size=beam_size)
    print('Inference from {} to {}'.format(src_lang, tgt_lang))
    for i, src in enumerate(examples[src_lang][:n_examples]):

                
        print(i+1)
        print('Source :', src)
        if use_tokenizer:
            if src_tok_type == 'newmm':
                toks = word_tokenize(src, engine='newmm', keep_whitespace=False)
            elif src_tok_type == 'sentencepiece':
                toks = encode_bpe([src], lang=src_lang)[0].split(' ')
        else:
            toks = src.split(' ')
        print('tokens :', toks)
        
        
        src_indices = [src_dict.index(t) for t in toks]
        src_len = [len(src_indices)]

        src_indices_tensor = torch.LongTensor(src_indices).cuda().view(1, -1)
        src_len_tensor = torch.LongTensor(src_len).cuda().view(1, -1)

        print('src indices:', src_indices_tensor.size(), src_indices_tensor)
        print('src len    :', src_len_tensor.size(), src_len_tensor)

        sample = {
            'net_input': {
                    'src_tokens': src_indices_tensor, 'src_lengths': src_len_tensor,
            },
        }
        
        prefix_tokens = None
        hypos =  _generator.generate(models=models, sample=sample, prefix_tokens=prefix_tokens)
        num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)

        for j, hypo in enumerate(hypos[i][:N_BEST]):
            if j == 0:
                if remove_bpe is not None:
                    # Convert back to tokens for evaluation with unk replacement and/or without BPE
                    target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
              
                scorer.add(target_tokens, hypo_tokens)

    return scorer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--src_dict_path", type=str)
    parser.add_argument("--tgt_dict_path", type=str)
    parser.add_argument("--remove_bpe", type=str2bool)
    parser.add_argument("--use_tokenizer", type=str2bool)
    parser.add_argument("--tgt_tok_type", type=str, help="newmm, sentencepiece_opensubtitles")
    parser.add_argument("--src_tok_type", type=str, help="newmm, sentencepiece_opensubtitles")

    args = parser.parse_args()

    scorer = evaluate(model_path=args.model_path,
             src_dict_path=args.src_dict_path,
             tgt_dict_path=args.tgt_dict_path,
             beam_size=args.beam_size,
             remove_bpe=args.remove_bpe,
             src_tok_type=args.src_tok_type,
             tgt_tok_type=args.tgt_tok_type)

    print(scorer)