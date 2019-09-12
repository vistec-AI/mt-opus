# fairseq-preprocess --source-lang en --target-lang th \
#     --trainpref data/opensubtitles_tok/train \
#     --validpref data/opensubtitles_tok/valid \
#     --testpref data/opensubtitles_tok/test \
#     --destdir data/opensubtitles_bin

fairseq-train \
    data/opensubtitles_bin \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 2048 \
    --bpe sentencepiece \
    --memory-efficient-fp16
    --save-dir data/opensubtitles_model/transformers
    
fairseq-generate data/opensubtitles_bin \
    --path data/opensubtitles_model/transformers/checkpoint_best.pt \
    --remove-bpe --beam 5 