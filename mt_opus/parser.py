import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def init_run_and_eval_parser():

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

    return parser