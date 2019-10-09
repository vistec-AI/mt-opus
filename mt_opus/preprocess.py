import re
import html
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


class SentencePairFoundRepeatedText(SentencePairRule):

    def __init__(self):
        super().__init__()

    @staticmethod    
    def test(sentence_pair):
        """
            sentence_pair Tuple[str, str] -- sentence pair
        """
        src, tgt = sentence_pair

        if tgt in src:
            return True

        if src in tgt:
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
        sentence = sentence.replace('\\"', '"')
        sentence = sentence.replace("\\'", "'")
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
    


    
class ReplaceDashInSentence(ReplaceRule):
    
    def __init__(self):
        super().__init__()
        
    @staticmethod
    def test(sentence, lang):
        if re.search(r"^\s*\-\s*", sentence):
            return True
        if re.search(r"\s*\-\s*$", sentence):
            return True
        return False

    @staticmethod
    def replace(sentence, lang):
        sentence = re.sub(r"^\s*\-\s*", '', sentence) # "-" found at the start
        sentence = re.sub(r"\s*\-\s*$", '', sentence) # "-" found at the end

        return sentence