#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""
import json
import argparse
import os
import pathlib
from functools import partial
from typing import List, Dict
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader

import fairseq

from fairseq.models.transformer import (
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS
) 

from fairseq import checkpoint_utils, options, tasks, bleu, options, progress_bar, utils
from fairseq.tasks import FairseqTask
from fairseq.tasks.translation import load_langpair_dataset
from fairseq.sequence_generator import SequenceGenerator
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.data import (
    Dictionary,
    FairseqDataset,
    LanguagePairDataset,
    data_utils,
    iterators
)


import sentencepiece as spm

from pythainlp.tokenize import word_tokenize
from tqdm import tqdm_notebook, tqdm

from mt_opus.dataset.wang import WangDataset
from utils import str2bool


_pythainlp_tokenize = partial(word_tokenize, engine="newmm", keep_whitespace=False)

DATASET_NAMES = ["wang"]
BLEU_ORDER = 4
N_BEST = 5

def get_batch_iterator(
    dataset: FairseqDataset,
    args,
    epoch: int = 0,
    seed: int = 1,
    max_positions = None,
    num_workers: int = 4,
    shard_id: int = 0,
    num_shards: int = 2,
):

    assert isinstance(dataset, FairseqDataset)
    # initialize the dataset with the correct starting epoch
    dataset.set_epoch(epoch)
    # get indices ordered by example size
    with data_utils.numpy_seed(seed):
        indices = dataset.ordered_indices()

    # filter examples that are too large
    if max_positions is not None:
        indices = data_utils.filter_by_size(
            indices, dataset, max_positions, raise_exception=(not args.ignore_invalid_inputs),
        )
    # create mini-batches with given size constraints
    batch_sampler = data_utils.batch_by_size(
        indices, dataset.num_tokens,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        required_batch_size_multiple=1,
    )

    epoch_iter = iterators.EpochBatchIterator(
        dataset=dataset,
        collate_fn=dataset.collater,
        batch_sampler=batch_sampler,
        seed=1,
        num_shards=num_shards,
        shard_id=shard_id,
        num_workers=num_workers,
        epoch=epoch,
    ).next_epoch_itr(shuffle=False)

    return epoch_iter


def log_result(path, result_str, mode="a"):
    # append Open for writing.  The file is created if it does not exist.  
    f = open(path, mode=mode)
    f.write(result_str)
    f.close()


def build_generator(tgt_dict, args):

    generator = SequenceGenerator(
                    tgt_dict=tgt_dict,
                    beam_size=getattr(args, 'beam', 5),
                    max_len_a=getattr(args, 'max_len_a', 0),
                    max_len_b=getattr(args, 'max_len_b', 200),
                    min_len=getattr(args, 'min_len', 1),
                    normalize_scores=(not getattr(args, 'unnormalized', False)),
                    len_penalty=getattr(args, 'lenpen', 1),
                    unk_penalty=getattr(args, 'unkpen', 0),
                    sampling=getattr(args, 'sampling', False),
                    sampling_topk=getattr(args, 'sampling_topk', -1),
                    sampling_topp=getattr(args, 'sampling_topp', -1.0),
                    temperature=getattr(args, 'temperature', 1.),
                    diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
                    diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
                    match_source_len=getattr(args, 'match_source_len', False),
                    no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
                )

    return generator
    return _apply(sample)


def _run_inference(parser_args, models, iterator, generator, scorer, use_cuda, tgt_dict, tgt_dict_newmm, src_dict):
    
    gen_timer = StopwatchMeter()

    list_translation_results = []

    num_sentences = 0
    has_target = True
    with progress_bar.build_progress_bar(args=parser_args, iterator=iterator) as t:
        wps_meter = TimeMeter()
        # print("length of t", len(t))
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
  
            if 'net_input' not in sample:
                continue

            # print("Sample net_input", sample["net_input"])
            # print("Sample src_tokens", sample["net_input"]["src_tokens"])
            # print("Sample src_tokens.size()", sample["net_input"]["src_tokens"].size())

            # print("Sample src_lengths", sample["net_input"]["src_lengths"].size())
            gen_timer.start()
            hypos =  generator.generate(models, sample, prefix_tokens=None)      
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
            gen_timer.stop(num_generated_tokens)

            for i, sample_id in enumerate(sample['id'].tolist()):
                has_target = sample['target'] is not None

                # Remove padding
                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())

                # print("src_tokens", src_tokens)

                target_tokens = None
                if has_target:
                    target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict_newmm.pad()).int().cpu()

                src_str = src_dict.string(src_tokens, parser_args.remove_bpe)
                if has_target:
                    # enforce not remove bpe
                    target_str = tgt_dict_newmm.string(target_tokens, None, escape_unk=True)


                for j, hypo in enumerate(hypos[i][:1]):
                    
                    # print("hypo['tokens']", hypo['tokens'])
               
                    # hypo_tokens, hypo_str, _ = utils.post_process_prediction(
                    #         hypo_tokens=hypo['tokens'].int().cpu(),
                    #         src_str=src_str,
                    #         alignment=None,
                    #         align_dict=None,
                    #         tgt_dict=tgt_dict,
                    #         remove_bpe=parser_args.remove_bpe,
                    # 
                    # print(j, "hypo_str:", hypo_str)
                    # Score only the top hypothesis
                    if has_target and j == 0:
                        hypo_ids = hypo['tokens'].int().cpu()
                        hypo_str = tgt_dict.string(hypo_ids, parser_args.remove_bpe)

                        if parser_args.remove_bpe is not None:
                            hypos_toks = _pythainlp_tokenize(hypo_str)
                            hypo_str = " ".join(hypos_toks)
                            # print("hypos_toks", hypos_toks)

                            hypo_ids = [tgt_dict_newmm.index(t) for t in hypos_toks ]
                            hypo_ids.append(tgt_dict_newmm.eos())

                            hypo_toks_tensor = torch.IntTensor(hypo_ids)
                            
                        else: 
                            hypo_toks_tensor = hypo_ids

                        list_translation_results.append({
                            "source_str": src_str,
                            "hypo_str": hypo_str,
                            "target_str": target_str })

                        scorer.add(target_tokens, hypo_toks_tensor)

            wps_meter.update(num_generated_tokens)
            t.log({'wps': round(wps_meter.avg)})
            num_sentences += sample['nsentences']
    
    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg)) 

    return scorer, list_translation_results


def _evaluate_per_epoch(epoch,
             dataset,
             model_path,
             src_dict,
             tgt_dict,
             tgt_dict_newmm,
             src_lang,
             tgt_lang,
             src_tok_type,
             tgt_tok_type,
             remove_bpe,
             use_cuda,
             use_tokenizer,
             parser_args,
             n_best=N_BEST) :

    
    # 1. Load model
    print("INFO: Load model from {}".format(model_path))

    state = checkpoint_utils.load_checkpoint_to_cpu(model_path)
    args = vars(state['args'])
    args['data'] = os.path.join(parser_args.data_prefix, args['data'])
 
    models, args, task = checkpoint_utils.load_model_ensemble_and_task([model_path], arg_overrides=args, task=None)
    # print("args", args)
    # print('task', task)

    # 2. Optimize ensemble for generation
    if parser_args.use_cuda:
        num_shards = torch.cuda.device_count()

    # set device if use cuda
    
        for model in models:
            model.make_generation_fast_(
                beamable_mm_beam_size=None if parser_args.beam == None else parser_args.beam,
                need_attn=False,
            )
            if use_cuda:
                model.cuda()

        # print(models)
        # 3. Acquite Batch iterator
        print("INFO: Acquire Batch iterator")
        _iterator = get_batch_iterator(dataset,
                                        parser_args,
                                        epoch=epoch,
                                        seed=1,
                                        max_positions=None,
                                        num_workers=4,
                                        num_shards=num_shards,
                                        shard_id=parser_args.gpu)
        # print("epoch_iter", iterator)
        # print('length of vocab size,', len(tgt_dict))

        # print('tgt vocab last 10 sumbols', tgt_dict[-1])
        _generator = build_generator(tgt_dict=tgt_dict, args=parser_args)


        # 5. Initiate scorer
        # scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
        _scorer = bleu.Scorer(tgt_dict_newmm.pad(), tgt_dict_newmm.eos(), tgt_dict_newmm.unk())


        scorer, list_translation_results = _run_inference(parser_args=parser_args,
                                                    models=models,
                                                    iterator=_iterator,
                                                    generator=_generator,
                                                    scorer=_scorer,
                                                    use_cuda=use_cuda,
                                                    tgt_dict=tgt_dict,
                                                    tgt_dict_newmm=tgt_dict_newmm,
                                                    src_dict=src_dict)
   

def evaluate(args):
    
    # intit tokenizer
    _pythainlp_tokenize = partial(word_tokenize, engine="newmm", keep_whitespace=False)

    bpe_model_opensubtitles = spm.SentencePieceProcessor()
    bpe_model_opensubtitles.Load(args.bpe_model_path)
    _sentencepiece_tokenize = partial(bpe_model_opensubtitles.EncodeAsPieces)

    device = torch.device("cuda" if args.use_cuda else "cpu")

    print("Selected device (use_cuda={}): {}".format(args.use_cuda, device))

    if args.dataset_name == "wang":

        src_tokenize = _pythainlp_tokenize if args.src_tok_type == "newmm" else _sentencepiece_tokenize
        tgt_tokenize = _pythainlp_tokenize # encode newmm

        dataset = WangDataset.from_text(path_to_text_file=args.examples_path,
                                        number_of_lines=args.n_examples,
                                        src_lang=args.src_lang,
                                        tgt_lang=args.tgt_lang,
                                        src_tokenize=src_tokenize,
                                        tgt_tokenize=tgt_tokenize)
       
        src_dict = Dictionary()
        src_dict.add_from_file(args.src_dict_path)
        src_dict.finalize()

        tgt_dict = Dictionary()
        tgt_dict.add_from_file(args.tgt_dict_path)

        tgt_dict_newmm = Dictionary()
        tgt_dict_newmm.add_from_file(args.tgt_dict_path_newmm)

        src_dataset, src_lengths, tgt_dataset, tgt_lengths = dataset.get_language_pair_datasets(
                                                                args.src_lang,
                                                                args.tgt_lang,
                                                                src_dict,
                                                                tgt_dict_newmm)

    
        
        # print("src_dataset", src_dataset[0:2])
        # print("tgt_dataset", tgt_dataset[0:2])
        # print("dataset.sent_th", dataset.sentence_pairs["th"])
        # print("dataset.sent_en", dataset.sentence_pairs["en"])

        lang_pair_dataset = LanguagePairDataset(
                                src=src_dataset,
                                src_sizes=src_lengths,
                                tgt=tgt_dataset,                               
                                tgt_sizes=tgt_lengths,
                                src_dict=src_dict,
                                tgt_dict=tgt_dict_newmm,
                                left_pad_source=False,
                                left_pad_target=False,
                                max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS,
                                max_target_positions=DEFAULT_MAX_TARGET_POSITIONS,)
        
        print("LanguagePairDataset created.")
        # print(lang_pair_dataset[0])
        # print(lang_pair_dataset[0]["source"][0])
        # exit()
    else:
        print("Argument dataset_name is invalid (please only specify a name in this list: {})".format(DATASET_NAMES))
        exit()

    # model_name = Path(model_path).stem
    print(args)
    print('\nStart evaluation.')

    pathlib.Path(args.result_dir).mkdir(parents=True, exist_ok=True) 
    result_blue_score_filename = 'translation_bleu_score.' + args.model_dir.split('/models/')[1].replace("/", ".") + ".dataset_name-{}.n_examples-{}.csv".format(args.dataset_name, args.n_examples)
    # Create Directory if not exists
    
    result_bleu_score_path = os.path.join(args.result_dir, result_blue_score_filename)
    result_str_header = "epoch,blue,p1,p2,p3,p4,bp,ratio,syslen,reflen\n"

    print("INFO: Log the result header to :{}".format(result_bleu_score_path))

    log_result(result_bleu_score_path, result_str_header, mode="w")

    for epoch in range(1, args.n_epochs + 1):

        print("Epoch Number:", epoch)
        model_path = os.path.join(args.model_dir, "checkpoint{}.pt".format(epoch))

        scorer, list_translation_results = _evaluate_per_epoch(
                                            epoch=epoch - 1,
                                            parser_args=args,
                                            dataset=lang_pair_dataset,
                                            use_cuda=args.use_cuda,
                                            model_path=model_path,
                                            src_dict=src_dict,
                                            tgt_dict=tgt_dict,
                                            tgt_dict_newmm=tgt_dict_newmm,
                                            src_lang=args.src_lang,
                                            tgt_lang=args.tgt_lang,
                                            use_tokenizer=args.use_tokenizer,
                                            remove_bpe=args.remove_bpe,
                                            src_tok_type=args.src_tok_type,
                                            tgt_tok_type=args.tgt_tok_type)
        
        print("\n\nBLEU Score:")
        print(scorer.result_string())


        print("INFO: Log the result of epoch {} to :{}".format(epoch, result_bleu_score_path))
        order = BLEU_ORDER    
        bleup = [p * 100 for p in scorer.precision()[:order]]
        result_str = result_str_header = "{},{:2.2f},{:2.1f},{:2.1f},{:2.1f},{:2.1f},{:.1f},{:.1f},{},{}".format(
                                            epoch,
                                            scorer.score(order=order),
                                            *bleup,
                                            scorer.brevity(),
                                            scorer.stat.predlen/scorer.stat.reflen,
                                            scorer.stat.predlen,
                                            scorer.stat.reflen)
                                          
        if epoch != args.n_epochs:
            result_str += "\n"
        log_result(result_bleu_score_path, result_str)
     
        # Log translation results:
        result_translation_filename = 'translation_result.epoch-{}_'.format(epoch) + args.model_dir.split('/models/')[1].replace("/", ".") + "dataset_name-{}.n_examples-{}.json".format(args.dataset_name, args.n_examples)
        result_translation_path = os.path.join(args.result_dir, result_translation_filename)
        print("INFO: Write the translation result to :{}".format(result_translation_path))

        with open(result_translation_path, "w", encoding="utf-8") as f:
            json.dump(obj={ "data": list_translation_results }, fp=f, ensure_ascii=False, indent=4)

    # print(list_translation_results)
    print('\Done evaluation.')

    return device

def plot_bleu_score(csv_file_path):
    """Plot bleu score (y) for each epoch (x) and save as png"""

    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_name", type=str, default="wang", help="Name of test dataset (e.g `wang`)")
    parser.add_argument("--examples_path", type=str, help="Path to the file storing dataset withno language id suffix (e.g `data/wang/wang.sent`)")
    parser.add_argument("--n_examples", type=int, default=None)
    parser.add_argument("--data_prefix", type=str, default="/storage-mt")
    parser.add_argument("--bpe_model_path", type=str, default="./data/sentencepiece_models/spm.opensubtitles.v2.model")
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--src_lang", type=str)
    parser.add_argument("--tgt_lang", type=str)
    parser.add_argument("--src_dict_path", type=str)
    parser.add_argument("--tgt_dict_path", type=str)
    parser.add_argument("--tgt_dict_path_newmm", type=str)
    parser.add_argument("--result_dir", type=str, default="./results/translation")

    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--max_sentences", type=int, default=100)
    parser.add_argument("--ignore_invalid_inputs", type=str2bool, default=False)
    parser.add_argument("--no_beamable_mm", type=str2bool, default=False)
    parser.add_argument("--log_format", type=str, default="simple")
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--no_progress_bar", type=str2bool, default=False)

    parser.add_argument("--tensorboard_logdir", type=str, default=None)
    parser.add_argument("--tbmf_wrapper", type=str, default=None)

    parser.add_argument("--remove_bpe", type=str, default=None, help="If target token type is SentencePiece, specify this argument as `sentencepiece` if not specigy is as`None`")
    parser.add_argument("--use_tokenizer", type=str2bool, help="Either to use the tokenizer (newmm, sentencepiece) to pretokenize source, target sentences before feed to the NMT model")
    parser.add_argument("--use_cuda", type=str2bool, help="Either to use GPU or not", default=False)
    parser.add_argument("--gpu", type=int, help="GPU ID", default=0)

    parser.add_argument("--tgt_tok_type", type=str, help="Either newmm, or sentencepiece_opensubtitles")
    parser.add_argument("--src_tok_type", type=str, help="Either newmm, sentencepiece_opensubtitles")

    args = parser.parse_args()
    
    evaluate(args)