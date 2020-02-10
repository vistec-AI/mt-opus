# mt-opus
English-Thai Machine Translation with OPUS data

## Data
We used 9 datasets from [OPUS](http://opus.nlpl.eu/index.php) to train and validate our models within and across domains (total 5.4M sentence pairs; 68.8M English tokens and 53.1M Thai tokens).

| datasets | nb_sent | en_tok | th_tok | description | reference |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|--------|--------|------------------------------------|-----------|
| [OpenSubtitles   v2018](http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-th.txt.zip) | 3.5M | 28.4M | 7.8M | crowdsourced subtitles | [1] |
| [JW300 v1](http://opus.nlpl.eu/JW300-v1.php)   [en](https://object.pouta.csc.fi/OPUS-JW300/v1/raw/en.zip)   [th](https://object.pouta.csc.fi/OPUS-JW300/v1/raw/th.zip) | 0.8M | 14.9M | 34.6M | Jehovah's Witness site | [2], [3] |
| [GNOME v1](https://object.pouta.csc.fi/OPUS-GNOME/v1/moses/en-th.txt.zip) | 0.5M | 2.3M | 3.5M | GNOME documentation | [2] |
| [QED v2.0a](https://object.pouta.csc.fi/OPUS-QED/v2.0a/moses/en-th.txt.zip) | 0.3M | 4.7M | 1.2M | crowdsourced educational subtitles | [2] |
| [bible-uedin v1](https://object.pouta.csc.fi/OPUS-bible-uedin/v1/moses/en-th.txt.zip) | 0.1M | 3.6M | 2.1M | the Bible | [2], [4] |
| [Tanzil v1](https://object.pouta.csc.fi/OPUS-Tanzil/v1/moses/en-th.txt.zip) | 93.5k | 2.8M | 3.4M | the Quran | [2] |
| [KDE4 v2](https://object.pouta.csc.fi/OPUS-KDE4/v2/moses/en-th.txt.zip) | 92.0k | 0.5M | 0.2M | KDE4 documentation | [2] |
| [Ubuntu v14.10](https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/moses/en-th.txt.zip) | 46.6k | 0.4M | 0.2M | Ubuntu documentation | [2] |
| [Tatoeba v20190709](https://object.pouta.csc.fi/OPUS-Tatoeba/v20190709/moses/en-th.txt.zip) | 1.1k | 6k | 1.7k | crowdsourced translations | [2] |

## Models

## Results

# References
* [1] P. Lison and J. Tiedemann, 2016, OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV Subtitles. In Proceedings of the 10th International Conference on Language Resources and Evaluation (LREC 2016)
* [2] J. Tiedemann, 2012, Parallel Data, Tools and Interfaces in OPUS. In Proceedings of the 8th International Conference on Language Resources and Evaluation (LREC 2012)
* [3]  Željko Agić, Ivan Vulić: "JW300: A Wide-Coverage Parallel Corpus for Low-Resource Languages", In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL), 2019. Acknowledge also OPUS by citing the following article: J. Tiedemann, 2012, Parallel Data, Tools and Interfaces in OPUS. In Proceedings of the 8th International Conference on Language Resources and Evaluation (LREC 2012)
* [4] A massively parallel corpus: the Bible in 100 languages, Christos Christodoulopoulos and Mark Steedman, *Language Resources and Evaluation*, 49
