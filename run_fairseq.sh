fairseq-preprocess --source-lang en --target-lang th \
    --trainpref data/opensubtitles_tok/train \
    --validpref data/opensubtitles_tok/valid \
    --testpref data/opensubtitles_tok/test \
    --destdir data/opensubtitles_bin

fairseq-train \
    data/opensubtitles_bin \
    --arch transformer_iwslt_de_en --max-epoch 10 \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 2048 \
    --bpe sentencepiece \
    --memory-efficient-fp16
    
fairseq-generate data/opensubtitles_bin \
    --path checkpoints/checkpoint_best.pt \
    --remove-bpe --beam 5 --max-tokens 2048

# | Translated 328154 sentences (2773243 tokens) in 1517.1s (216.31 sentences/s, 1828.01 tokens/s)
# | Generate test with beam=5: BLEU4 = 10.80, 36.6/15.3/7.2/3.4 (BP=1.000, ratio=1.029, syslen=2445089, reflen=2376306)