import re
import html

from collections import Counter
from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sentencepiece as spm
import tf_sentencepiece 

from pythainlp.tokenize import word_tokenize
from pythainlp.util.normalize import normalize as thai_text_normalize
from pythainlp.util.normalize import _NORMALIZE_RULE2, _NORMALIZE_RULE1
from pythainlp.util.thai import countthai

class BaseRule:

    @staticmethod    
    def test(sentence, lang):
        pass

class SentencePairRule:

    @staticmethod    
    def test(sentence_pair):
        pass

        
class ReplaceRule(BaseRule):   
    
    def __init__(self):
        super().__init__()
    
    @staticmethod 
    def replace(sentence, lang):
        pass

class SentenceContainsOnlyAsterisk(BaseRule):

    def __init__(self):
        super().__init__()

    @staticmethod    
    def test(sentence, lang):
      
        if re.search(r"^[\s]*\*{1,}[\s]*$", sentence):
            return True
    
        return False


class SentencePairFoundRepeatedText(SentencePairRule):

    def __init__(self):
        super().__init__()

    @staticmethod    
    def test(sentence_pair):
        """
            sentence_pair Tuple[str, str] -- sentence pair
        """
        src, tgt = sentence_pair

        if len(tgt) == 0 or len(src) == 0:
            return True

        if tgt in src or src in tgt:
            return True
        return False

###



class SentencePairUSESimilarityLessThanThreashold(SentencePairRule):



    def __init__(self, threshold = 0.85):
        super().__init__()
        self.threshold = threshold

        self._graph = tf.Graph()
        with self._graph.as_default():
            self.text_input = tf.placeholder(dtype=tf.string, shape=[None])
            embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-multilingual/1")
            self.embedded_text = embed(self.text_input)
            self._init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        self._graph.finalize()

    
    def _compute_similarity(self, src_sent, tgt_sent):
        """
            Calculate sentence similarity based on Google Universal Sentence Encoder (Multilingual Large)
        """
        session = tf.Session(graph=self._graph)
        session.run(self._init_op)
        
        src_result = session.run(self.embedded_text, feed_dict={self.text_input: [src_sent]})
        tgt_result = session.run(self.embedded_text, feed_dict={self.text_input: [tgt_sent]})
        
        return np.inner(src_result, tgt_result).reshape(1)[0]


    def test(self, sentence_pair):
        """
            sentence_pair Tuple[str, str] -- sentence pair
        """
        src, tgt = sentence_pair
        
        similarity = self._compute_similarity(src, tgt)

        if similarity < self.threshold:
            return True

        return False


class SentencePairTokenLengthsDifferGreaterThreshold(SentencePairRule):

    def __init__(self, threshold = 0.85):
        super().__init__()
        self.threshold = threshold
        self._tokenize = partial(word_tokenize, engine="newmm", keep_whitespace=False)
  
    def test(self, sentence_pair):
        """
            sentence_pair Tuple[str, str] -- sentence pair
        """
        src, tgt = sentence_pair
        
        src_toks, tgt_toks = self._tokenize(src), self._tokenize(tgt)
        src_toks_len, tgt_toks_len = len(src_toks), len(tgt_toks)
        diff = abs(src_toks_len - tgt_toks_len)
        diff_ratio =  diff / (max(src_toks_len, tgt_toks_len))

        if diff_ratio > self.threshold:
            return True

        return False

class UnescapeString(ReplaceRule):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _unescape_string(text):
        return html.unescape(text)
    
    @staticmethod
    def test(sentence, lang):
        return True

    @staticmethod
    def replace(sentence, lang):
        sentence = re.sub(r'[\\]+"', '"', sentence)
        sentence = re.sub(r"[\\]+'", "'", sentence)
        return UnescapeString._unescape_string(sentence)


class RemoveHashtagInSentence(ReplaceRule):
    
    def __init__(self):
        super().__init__()


    @staticmethod
    def test(sentence, lang):
        if re.match(r"^[\s\-]*#+", sentence):
            return True
        if re.search(r"[\s\-]*#$", sentence):
            return True
        if re.search(r"[\s\-]*#", sentence):
            return True
        return False
    
    @staticmethod
    def replace(sentence, lang):
        # replace hash tag at the start of sentence
        sentence = re.sub(r"^[\s\-]*#+", '', sentence).lstrip() 

        # replace hash tag at the end of sentence
        sentence = re.sub(r"#+[\s\-]*$", '', sentence).rstrip()

        # replace hash tag (/\-#/, /#/)in between the sentence
        sentence = re.sub(r"[\-]*[\s]*#", '', sentence)
        return sentence


class NormalizeThaiVowel(ReplaceRule):
    
    def __init__(self):
        super().__init__()

    @staticmethod
    def test(sentence, lang):
        if lang == 'th':
            _NORMALIZE_RULE2 = []
            if re.search(r'เเ', sentence):
                return True
            for rule in list(zip(_NORMALIZE_RULE1, _NORMALIZE_RULE1)):
                if re.search(rule[0].replace("t", "[่้๊๋]") + '+' + rule[1], sentence):
                    return True

        return False

    @staticmethod
    def replace(sentence, lang):
        if lang == "th":
            sentence = thai_text_normalize(sentence)
        return sentence


class SentenceLengthLessThanOrEqualToOne(BaseRule):
    def __init__(self):
        super().__init__()
        
    @staticmethod
    def test(sentence, lang):
        sentence = sentence.strip()
        if len(sentence) <= 1:
            return True
        return False

DEFAULT_UNWANTED_SYMBOLS = [
    b'\xc2\x99',
    b'\xc2\x84',
    b'\xe2\x80\x8b',
    b'\xc2\x94',
    b'\xc2\x96',
    b'\xc2\x93',
    b'\xc2\x85',
    b'\xc2\x97',
    b'\xe0\xb9\x82\xc2\x80',
    b'\xe2\x80\x8b',
    b'\xe2\x99\xaa',
]

class RemoveUnwantedSymbols(ReplaceRule):
    def __init__(self, symbol_list=DEFAULT_UNWANTED_SYMBOLS):
        """
         symbol_list [str, regex pattern]
       ` """
        super().__init__()
        self.symbol_list = symbol_list

    def test(self, sentence, lang):
        for symbol in self.symbol_list:
            if symbol in sentence.encode('utf-8'):
                return True
        return False

    def replace(self, sentence, lang):
        for symbol in self.symbol_list:
            sentence = sentence.encode('utf-8').replace(symbol, b'').decode('utf-8')
        return sentence

DEFAULT_UNWANTED_PATTERN =  [
    r"[\{]*[\s]*[\\]*[\s]*cHFFFFFF[\s]*[\}]*",
    r"\{\}",
    r"[\\]{1,}[\s]*[N|i1|NI]*",
    r"[\s]*[#|,]*8203;[\s]*",
    r"font color[\s]*=[\s]*\"#[\s]*[\d]{3,6}[\s]*\""
]

class RemoveUnwantedPattern(ReplaceRule):
    def __init__(self, pattern_list=DEFAULT_UNWANTED_PATTERN):
        """
         pattern_list [regex pattern]
        """
        super().__init__()
        self.pattern_list = pattern_list

    def test(self, sentence, lang):
        for pattern in self.pattern_list:
            if re.search(pattern, sentence):
                return True
        return False

    def replace(self, sentence, lang):
        for pattern in self.pattern_list:
            sentence = re.sub(pattern, '', sentence)
        return sentence

class SentenceContainsUnknownSymbol(BaseRule):
    def __init__(self):
        super().__init__()
        self.list_unknown_symbols = [ b'\x98\xc2',
                                      b'\xae\xc2', 
                                      b'\xb1\xc2',
                                      b'\xc2\xb7',
                                      b'\xc2\x8b',
                                      b'\xc3\x83']
    
    def test(self, sentence, lang):
        for symbol in self.list_unknown_symbols:
            if symbol in sentence.encode('utf-8'):
                return True
        return False



class SentenceContainsAdSymbol(BaseRule):
    def __init__(self):
        super().__init__()
        
    @staticmethod
    def test(sentence, lang):
        if '@' in sentence:
            return True
        return False


DEFAULT_LIST_OF_UNWANTED_PATTERN_THAI = [
    r'^\*เพลง\*$',
    r'^เพลง$',
    r'^[คำ]*บรรยาย[\s]*((ตาม)|(โดย))',
]
class ThaiSentenceContainsUnwantedPattern(BaseRule):
    def __init__(self, list_of_pattern=DEFAULT_LIST_OF_UNWANTED_PATTERN_THAI):
        super().__init__()
        self.list_of_pattern = list_of_pattern
        
    def test(self, sentence, lang):
        if lang == "th":
            for pattern in self.list_of_pattern:
                if re.compile(pattern).search(sentence):
                    return True
        return False

DEFAULT_LIST_OF_UNWANTED_PATTERN_ENGLISH = [
    r'(?i)(^subtitle[s]*)|(subtitle[s]*$)|(subtitles ((by)|(conformed by)|(downloaded from)))',
]
class EnglishSentenceContainsUnwantedPattern(BaseRule):
    def __init__(self, list_of_pattern=DEFAULT_LIST_OF_UNWANTED_PATTERN_ENGLISH):
        super().__init__()
        self.list_of_pattern = list_of_pattern
        
    def test(self, sentence, lang):
        if lang == "en":
            for pattern in self.list_of_pattern:
                if re.compile(pattern).search(sentence):
                    return True
        return False

class ThaiSentenceContainsNoThaiCharacters(BaseRule):
    def __init__(self):
        super().__init__()
        
    @staticmethod
    def test(sentence, lang):
        if lang == "th":
            if countthai(sentence, ignore_chars='') == 0.0:
                return True
        return False

class ThaiSentenceContainsNoThaiCharactersPattern(BaseRule):
    def __init__(self):
        super().__init__()
        
    @staticmethod
    def test(sentence, lang):
        if lang == "th":
            if b'\xe0\xb9\x82\xc2\x99\xe0\xb8\x8a' in sentence.encode('utf-8'):
                return True
            if re.search(r'^โช[\s]*[A-z]', sentence):
                return True
            if re.search(r'^โช\s\.', sentence):
                return True
            if re.search(r'โช\s*$', sentence):
                return True
            if re.search(r'^โช\sโช', sentence):
                return True
            if re.search(r'โ#[\d]{3,6};', sentence):
                return True
            if re.search(r'N#[\d]{3,6};', sentence):
                return True
            if re.search(r'#/N#', sentence):
                return True
            if re.search(r'ใใใ', sentence):
                return True
        return False

class RemoveFullStopInThaiSentence(ReplaceRule):

    def __init__(self):
        super().__init__()
    
    @staticmethod
    def test(sentence, lang):
        if lang == 'th':
            if re.search(r'\.$', sentence):
                return True
        return False

    @staticmethod
    def replace(sentence, lang):
        sentence = re.sub(r"\.$", '', sentence)
        return sentence
    

class RemoveColonInSentence(ReplaceRule):

    def __init__(self):
        super().__init__()
    
    @staticmethod
    def test(sentence, lang):
        if re.search(r"[\s]*:$", sentence):
            # Count: 3015 sentence pairs.
            return True
        return False

    @staticmethod
    def replace(sentence, lang):
        sentence = re.sub(r"[\s]*:$", '', sentence)
        return sentence

class RemoveSemiColonInSentence(ReplaceRule):

    def __init__(self):
        super().__init__()
    
    @staticmethod
    def test(sentence, lang):
        if re.search(r"[\s]*;$", sentence):
            # Count: 198 sentence pairs.
            return True
        return False

    @staticmethod
    def replace(sentence, lang):
        sentence = re.sub(r"[\s]*;$", '', sentence)
        return sentence
    

class FormatTime(ReplaceRule):

    def __init__(self):
        super().__init__()
    
    @staticmethod
    def test(sentence, lang):
        if re.search(r"\d\d:\s\d\d", sentence):
            # Count: 565 sentence pairs.
            return True
        return False

    @staticmethod
    def replace(sentence, lang):
        sentence = re.sub(r"(\d\d:)(\s)(\d\d)", lambda m: m.group(1) + m.group(3), sentence)
        return sentence
    
    
class ReplaceDashInSentence(ReplaceRule):
    
    def __init__(self):
        super().__init__()
        
    @staticmethod
    def test(sentence, lang):
        if re.search(r"^\s*\-\s*", sentence):
            return True
        if re.search(r"\s*\-\s*$", sentence):
            return True
        if re.search(r"([\-]{2,})[\s]+", sentence):
            return True
        if re.search(r"([\.\?\!A-z\u0E00-\u0E7F])([\-]{2,})([\.\?\!A-z\u0E00-\u0E7F])", sentence):
            return True
        if re.search(r"(\-\s){2,}", sentence):
            return True
        if re.search(r"[\s]+([\-]{2,})", sentence):
            return True
        if re.search(r"[\s]+\-[\s]+", sentence):
            return True
        if re.search(r"[\s]+([\-]{2,})", sentence): 
            return True
        if re.search(r"([\-]{2,})[\s]+", sentence): 
            return True
        if re.search(r"\-\s\-", sentence):
            return True
        return False

    @staticmethod
    def replace(sentence, lang):
        sentence = re.sub(r"^\s*[\-]{1,}\s*", '', sentence) # "-" found at the start
        sentence = re.sub(r"\s*[\-]{1,}\s*$", '', sentence) # "-" found at the end
        
        sentence = re.sub(r"[\s]+([\-]{2,})", ' ', sentence) # " --" -> " "
        sentence = re.sub(r"([\-]{2,})[\s]+", ' ', sentence) # "-- " -> " "
        sentence = re.sub(r"[\s]*(\-\s){2,}", ' ', sentence) # " - - - " -> " " or "- - - " -> " "

        sentence = re.sub(r"[\s]+\-[\s]+", ' ', sentence) # " - " -> " "
        
        sentence = re.sub(r"\-\s\-", ' ', sentence) # "- -" -> " "

        # substitute "\w\-{2,}\w" to "\w \w" where \w is any character (Thai and English)
        sentence = re.sub(r"([0-9\.\?\!A-z\u0E00-\u0E7F])([\-/]{2,})([0-9\.\?\!A-z\u0E00-\u0E7F])", lambda m: m.group(1) + ' ' + m.group(3), sentence)
        sentence = re.sub(r"([\s]+)([\-]{1,})([0-9\.\?\!A-z\u0E00-\u0E7F])", lambda m: ' ' + m.group(3), sentence)

        return sentence

class ReplaceAsteriskInSentence(ReplaceRule):
    
    def __init__(self):
        super().__init__()
        
    @staticmethod
    def test(sentence, lang):
        # "*" at the start of sentence
        if re.search(r"^[\s]*[\*]{1,}[\s]*", sentence):
            return True
        # "*" at the end of sentence
        if re.search(r"[\s]*[\*]{1,}[\s]*$", sentence):
            return True
        # " *{1,} " in the sentence
        if re.search(r"[\s]+[\*]{1,}[\s]+", sentence):
            return True

        if re.search(r"\s(\*\s){1,}", sentence):
            return True

        if re.search(r"^[\s]*(\*\s){1,}", sentence):
            return True

        if re.search(r"\*", sentence):
            return True

        return False

    @staticmethod
    def replace(sentence, lang):
        sentence = re.sub(r"^[\s]*[\*]{1,}[\s]*", '', sentence) # ^"* " -> "" or ^" ****" -> ""
        sentence = re.sub(r"[\s]*[\*]{1,}[\s]*$", '', sentence) # "* "$ -> "" or " ****"$ -> ""

        sentence = re.sub(r"[\s]+[\*]{1,}[\s]+", ' ', sentence) # " * " -> " "

        sentence = re.sub(r"\s(\*\s){1,}", ' ', sentence) # " * * " -> " "
        sentence = re.sub(r"^[\s]*(\*\s){1,}", '', sentence) # ^" * *" -> ""
        
        sentence = re.sub(r"\*", '', sentence) # "*" -> ""

        return sentence


class SentenceContainsTwoDashes(BaseRule):
     
    def __init__(self):
        super().__init__()
               
    @staticmethod
    def test(sentence, lang):
        # case: 
        # "\w\-\-\w" to "\w \w" where \w is any character (Thai and English)
        if re.search(r"([A-z\u0E00-\u0E7F])([\-]{2,})([A-z\u0E00-\u0E7F])", sentence):
            return True

        return False