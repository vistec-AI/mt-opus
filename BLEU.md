__BLEU score evaluation:__

Given the source and reference sentences that are not tokenized

1. Tokenize the source and reference sentences with `newmm` and stored them.
   For example:

    - TH sentence: `"ฉันไปโรงเรียน"` → `["ฉัน", "ไป", "โรง", "เรียน"]`

    - EN sentence: `"I go to school."` → `["I", "go", "to", "school", "."]`

2. Feed the untokenized source sentences to NMT models.

    - th → en
        - word → word
          1. Tokenize the source sentences with a word-level tokenizer (e.g. `newmm`).
          2. Feed tokenized text to NMT model.
          3. Concatenate the predicted translation with spaces.  
             For example, `["I", "go", "to", "my", "school."]` → `"I go to my school."`.
          4. Tokenize the concatenated text by newmm tokenizer (with the same configuration from (1)).
             For example, `"I go to my school."` → `["I", "go" ,"to", "my" "school", "."]`
          5. Use the tokenized predicted translation and compute BLEU score with the reference sentences tokenized from (1)

        - subword → word
          1. Tokenize the source sentences with a  subword-level tokenizer (e.g. `SentencePiece`).
          2. Feed tokenized text to NMT model.
          3. Concatenate the predicted translation with spaces.  
             For example, `["I", "go", "to", "my", "school."]` → `"I go to my school"`.
          4. Tokenize the concatenated text by `newmm` tokenizer (with the same configuration from (1)).
             For example, `"I go to my school."` → `["I", "go" ,"to", "my" "school", "."]`      
          5. Use the tokenized predicted translation and compute BLEU score with the reference sentences tokenized from (1)

        - word → subword
          1. Tokenize the source sentences with a word-level tokenizer (e.g. `newmm`).
          2. Feed tokenized text to NMT model.
          3. Apply the BPE removing operation to predicted translation 
          For example,`["_I", "_go", "_to", "_my", "_sch", "ool", "."]` to `"I go to my school."`.
          1. Tokenize the text after BPE was removed by `newmm` tokenizer (with the same configuration from (1)).
            For example, `"I go to my school."` → `["I", "go" ,"to", "my" "school", "."]`

          2. Use the tokenized predicted translation and compute BLEU score with the reference sentences tokenized from (1)

        - subword → subword
          1. Tokenize the source sentences with a subword-level tokenizer (e.g. `SentencePiece`).
          2. Feed tokenized text to NMT model.
          3. Apply the BPE removing operation to predicted translation 
             For example,`["_I", "_go", "_to", "_my", "_sch", "ool", "_"]` to `"I go to my school."`.
          4. Tokenize the text after BPE was removed by `newmm` tokenizer (with the same configuration from (1)).
             For example, `"I go to my school."` → `["I", "go" ,"to", "my" "school", "."]`
          5. Use the tokenized predicted translation and compute BLEU score with the reference sentences tokenized from (1)

    - en → th
        - word → word
          1. Tokenize the source sentences with a word-level tokenizer (e.g. `newmm`).
          2. Feed tokenized text to NMT model.
          3. Concatenate the predicted translation with spaces.  
             For example, `["ฉัน", "ไปที่", "โรงเรียน", "มัธยม"]` → `"ฉัน ไปที่ โรงเรียน มัธยม"`.
          4. Tokenize the concatenated text by newmm tokenizer (with the same configuration from (1)).
             For example, `"ฉัน ไปที่ โรงเรียน มัธยม"` → `["ฉัน", "ไปที่", "โรง", "เรียน", "มัธยม"]` .
          5. Use the tokenized predicted translation and compute BLEU score with the reference sentences tokenized from (1)

        - subword → word
          1. Tokenize the source sentences with a  subword-level tokenizer (e.g. `SentencePiece`).
          2. Feed tokenized text to NMT model.
          3. Concatenate the predicted translation with spaces.  
             For example, `["ฉัน", "ไปที่", "โรงเรียน", "มัธยม"]` → `"ฉัน ไปที่ โรงเรียน มัธยม"`.
          4. Tokenize the concatenated text by `newmm` tokenizer (with the same configuration from (1)).
             For example, `"ฉัน ไปที่ โรงเรียน มัธยม"` → `["ฉัน", "ไปที่", "โรง", "เรียน", "มัธยม"]` .
          5. Use the tokenized predicted translation and compute BLEU score with the reference sentences tokenized from (1)

        - word → subword
          1. Tokenize the source sentences with a word-level tokenizer (e.g. `newmm`).
          2. Feed tokenized text to NMT model.
          3. Apply the BPE removing operation to predicted translation.
             For example, `["_ฉัน", "ไป", "_ที่", "_โรงเรียน", "มัถยม"]` → `"ฉันไป ที่ โรงเรียนมัธยม"`)`
          4. Tokenize the concatenated text by `newmm` tokenizer (with the same configuration from (1)).
             For example, `"ฉันไป ที่ โรงเรียนมัธยม"`  → `["ฉัน", "ไป", "ที่", "โรง","เรียน", "มัธยม"]`)`
          5. Use the tokenized predicted translation and compute BLEU score with the reference sentences tokenized from (1)

        - subword → subword
          1. Tokenize the source sentences with a subword-level tokenizer (e.g. `SentencePiece`).
          2. Feed tokenized text to NMT model.
          3. Apply the BPE removing operation to predicted translation 
             For example, `["_ฉัน", "ไป", "_ที่", "_โรงเรียน", "มัถยม"]` → `"ฉันไป ที่ โรงเรียนมัธยม"`)`
          4. Tokenize the text after BPE was removed by `newmm` tokenizer (with the same configuration from (1)).
             For example, `"ฉันไป ที่ โรงเรียนมัธยม"`  → `["ฉัน", "ไป", "ที่", "โรง","เรียน", "มัธยม"]`)`
          5. Use the tokenized predicted translation and compute BLEU score with the reference sentences tokenized from (1)

### Example:

__th → en__

   - newmm → newmm

      - src sentence: `ฉันโทรไปที่ร้านไก่กระสุน วันนี้`

      - src sentence tokenized: `['ฉัน', 'โทร', 'ไป', 'ที่', 'ร้าน', 'ไก่', 'กระสุน', 'วันนี้']`

      - predicted tokens (before retokenize): `['I', 'call', 'to', 'the', 'Bullet', 'chicken', 'shop', 'today', '.']`
      - predicted sentence (aftrer concatenation): `I call to the Bullet chicken shop today .`

      - predicted_tokens (after retokenize): `['I', 'call', 'to', 'the', 'Bullet', 'chicken', 'shop', 'today', '.']`

      - score = (bleu, precisions, bp, ratio, translation_length, reference_length) 
   	         = (0.43443712531357925, [0.7777777777777778, 0.5, 0.42857142857142855, 0.3333333333333333], 0.8948393168143697, 0.9, 9, 10)

   - newmm → sentencepiece

      - src sentence: `ฉันโทรไปที่ร้านไก่กระสุน วันนี้`

      - src sentence tokenized: `['ฉัน', 'โทร', 'ไป', 'ที่', 'ร้าน', 'ไก่', 'กระสุน', 'วันนี้']`

      - predicted tokens (before retokenize): `['_I', '_call', '_to', '_the', '_Bullet', '_chicken', '_shop', '_today.']`
      - predicted sentence (aftrer remobe bpe): ` I call to the Bullet chicken shop today.`

      - predicted_tokens (after retokenize): `['I', 'call', 'to', 'the', 'Bullet', 'chicken', 'shop', 'today', '.']`

      - score = (bleu, precisions, bp, ratio, translation_length, reference_length) 
   	         = (0.43443712531357925, [0.7777777777777778, 0.5, 0.42857142857142855, 0.3333333333333333], 0.8948393168143697, 0.9, 9, 10)

   - sentencepiece → newmm

      - src sentence: `ฉันโทรไปที่ร้านไก่กระสุน วันนี้`

      - src sentence tokenized: `['▁ฉัน โทร ไปที่ ร้าน ไก่ กระสุน ▁วัน นี้']`

      - predicted tokens (before retokenize): `['I', 'call', 'to', 'the', 'Bullet', 'chicken', 'shop', 'today', '.']`
      - predicted sentence (aftrer concatenation): `I call to the Bullet chicken shop today .`

      - predicted_tokens (after retokenize): `['I', 'call', 'to', 'the', 'Bullet', 'chicken', 'shop', 'today', '.']`

      - score = (bleu, precisions, bp, ratio, translation_length, reference_length) 
               = (0.43443712531357925, [0.7777777777777778, 0.5, 0.42857142857142855, 0.3333333333333333], 0.8948393168143697, 0.9, 9, 10)

   - sentencepiece → sentencepiece
 
      - src sentence: `ฉันโทรไปที่ร้านไก่กระสุน วันนี้`

      - src sentence tokenized: `['▁ฉัน โทร ไปที่ ร้าน ไก่ กระสุน ▁วัน นี้']`

      - predicted tokens (before retokenize): `['_I', '_call', '_to', '_the', '_Bullet', '_chicken', '_shop', '_today.']`
      - predicted sentence (aftrer remobe bpe): ` I call to the Bullet chicken shop today.`

      - predicted_tokens (after retokenize): `['I', 'call', 'to', 'the', 'Bullet', 'chicken', 'shop', 'today', '.']`

      - score = (bleu, precisions, bp, ratio, translation_length, reference_length) 
   	         = (0.43443712531357925, [0.7777777777777778, 0.5, 0.42857142857142855, 0.3333333333333333], 0.8948393168143697, 0.9, 9, 10)


__en → th__

   - newmm → newmm

      - src sentence: `Today, I call to the Bullet Chicken shop.`

      - src sentence tokenized: `['Today', ',', 'I', 'call', 'to', 'the', 'Bullet', 'Chicken', 'shop', '.']`

      - predicted tokens (before retokenize): `['วันนี้', 'ฉัน', 'โทร', 'ไป', 'ที่', 'ร้าน', 'ไก่', 'กระสุน']`
      - predicted sentence (aftrer concatenation): `วันนี้ ฉัน โทร ไป ที่ ร้าน ไก่ กระสุน`

      - predicted_tokens (after retokenize): `['วันนี้', 'ฉัน', 'โทร', 'ไป', 'ที่', 'ร้าน', 'ไก่', 'กระสุน']`

      - score = (bleu, precisions, bp, ratio, translation_length, reference_length) 
   	         = (0.8694417438899829, [1.0, 0.8571428571428571, 0.8333333333333334, 0.8], 1.0, 1.0, 8, 8)

   - newmm → sentencepiece

      - src sentence: `Today, I call to the Bullet Chicken shop.`

      - src sentence tokenized: `['Today', ',', 'I', 'call', 'to', 'the', 'Bullet', 'Chicken', 'shop', '.']`

      - predicted tokens (before retokenize): `['_วันนี้', '_ฉัน', 'โทร', 'ไป', 'ที่', '_ร้าน', 'ไก่', 'กระ', '_สุน']`
      - predicted sentence (aftrer remobe bpe): ` วันนี้ ฉันโทรไปที่ ร้านไก่กระ สุน`

      - predicted_tokens (after retokenize): `['วันนี้', 'ฉัน', 'โทร', 'ไป', 'ที่', 'ร้าน', 'ไก่', 'กระ', 'สุ', 'น']`

      - score = (bleu, precisions, bp, ratio, translation_length, reference_length) 
   	         = (0.537284965911771, [0.7, 0.5555555555555556, 0.5, 0.42857142857142855], 1.0, 1.25, 10, 8)

   - sentencepiece → newmm

      - src sentence: `Today, I call to the Bullet Chicken shop.`

      - src sentence tokenized: `['▁today , ▁i ▁call ▁to ▁the ▁bullet ▁chicken ▁shop .']`

      - predicted tokens (before retokenize): `['วันนี้', 'ฉัน', 'โทร', 'ไป', 'ที่', 'ร้าน', 'ไก่', 'กระสุน']`
      - predicted sentence (aftrer concatenation): `วันนี้ ฉัน โทร ไป ที่ ร้าน ไก่ กระสุน`

      - predicted_tokens (after retokenize): `['วันนี้', 'ฉัน', 'โทร', 'ไป', 'ที่', 'ร้าน', 'ไก่', 'กระสุน']`

      - score = (bleu, precisions, bp, ratio, translation_length, reference_length) 
               = (0.8694417438899829, [1.0, 0.8571428571428571, 0.8333333333333334, 0.8], 1.0, 1.0, 8, 8)

   - sentencepiece → sentencepiece

      - src sentence: `Today, I call to the Bullet Chicken shop.`

      - src sentence tokenized: `['▁today , ▁i ▁call ▁to ▁the ▁bullet ▁chicken ▁shop .']`

      - predicted tokens (before retokenize): `['_วันนี้', '_ฉัน', 'โทร', 'ไป', 'ที่', '_ร้าน', 'ไก่', 'กระ', '_สุน']`
      - predicted sentence (aftrer remobe bpe): ` วันนี้ ฉันโทรไปที่ ร้านไก่กระ สุน`

      - predicted_tokens (after retokenize): `['วันนี้', 'ฉัน', 'โทร', 'ไป', 'ที่', 'ร้าน', 'ไก่', 'กระ', 'สุ', 'น']`

      - score = (bleu, precisions, bp, ratio, translation_length, reference_length) 
   	         = (0.537284965911771, [0.7, 0.5555555555555556, 0.5, 0.42857142857142855], 1.0, 1.25, 10, 8)