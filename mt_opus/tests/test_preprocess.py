# coding: utf-8
from __future__ import unicode_literals

import pytest
from mt_opus.preprocess import (
    UnescapeString,
    RemoveHashtagInSentence,
    NormalizeThaiVowel,
    SentenceLengthLessThanOrEqualToOne,
    SentenceContainsUnknownSymbol,
    SentenceContainsAdSymbol,
    ThaiSentenceContainsNoThaiCharacters,
    ThaiSentenceContainsNoThaiCharactersPattern,
    RemoveFullStopInThaiSentence,
    ReplaceDashInSentence,
    RemoveUnwantedSymbols,
    RemoveUnwantedPattern,
    SentencePairFoundRepeatedText,
)

@pytest.mark.parametrize(
     "string, tested",
    [
        ("Hello world", True),
        ('สวัสดี \\" ซาวาน', True),
        ("สวัสดี \\' ซาวาน", True),
        ("สวัสดี \\'\\' ซาวาน", True),
        ("สวัสดี ซาวาน &amp; นกกา", True),
    ]
)
def test_UnescapeString_test(string, tested):
    rule = UnescapeString()
    output_tested = rule.test(string, lang="th")
    assert tested == output_tested

    output_tested = rule.test(string, lang="en")
    assert tested == output_tested

@pytest.mark.parametrize(
    "string, replaced",
    [
        ("Hello world", "Hello world"),
        ('สวัสดี \\" ซาวาน', 'สวัสดี " ซาวาน'),
        ("สวัสดี \\' ซาวาน", "สวัสดี ' ซาวาน"),
        ("สวัสดี \\'\\' ซาวาน", "สวัสดี '' ซาวาน"),
        ("สวัสดี ซาวาน &amp; นกกา", "สวัสดี ซาวาน & นกกา"),
    ]
)
def test_UnescapeString_replace(string, replaced):
    rule = UnescapeString()
    output_replaced = rule.replace(string, lang="th")
    assert replaced == output_replaced

    output_replaced = rule.replace(string, lang="en")
    assert replaced == output_replaced

@pytest.mark.parametrize(
    "string, tested",
    [
        ('เวลา กาลเวลา Tri-gram @Fl.9', False),
        ('## I am already die inside# ', True),
        ('# I am already die inside# ', True), 
        (' # I am already die inside #', True),
        ('### don\'t make me die twice ##', True),
        ('You got troubles -# And I got \'em too', True),
        ('You got troubles # And I got \'em too', True),
        ('You got troubles# And I got \'em too', True),

    ]
)
def test_RemoveHashtagInSentence_test(string, tested):
    rule = RemoveHashtagInSentence()
    output_tested = rule.test(string, lang="th")
    assert tested == output_tested

    output_tested = rule.test(string, lang="en")
    assert tested == output_tested

@pytest.mark.parametrize(
    "string, replaced",
    [
        ('เวลา กาลเวลา Tri-gram @Fl.9', "เวลา กาลเวลา Tri-gram @Fl.9"), # normal
        # hashtag at the start and end of sentence
        ('## I am already die inside# ', 'I am already die inside'),
        ('# I am already die inside# ', 'I am already die inside'), 
        (' # I am already die inside #', 'I am already die inside'),
        ('### don\'t make me die twice ##', 'don\'t make me die twice'),
        ('# You wanna dance?', 'You wanna dance?'),
        ('#อยากดิ้นกันมั้ย?', 'อยากดิ้นกันมั้ย?'),
        ('-# You got troubles # -# And I got \'em too #', 'You got troubles  And I got \'em too'),
        ('-# ฉันให้ความเคารพ # - # นับถือ #', 'ฉันให้ความเคารพ  นับถือ'),
        ('ฉันให้ความเคารพ - นับถือ #', 'ฉันให้ความเคารพ - นับถือ'),
    ]
)

def test_RemoveHashtagInSentence_replace(string, replaced):
    rule = RemoveHashtagInSentence()
    output_replaced = rule.replace(string, lang="th")
    assert replaced == output_replaced

    output_replaced = rule.replace(string, lang="en")
    assert replaced == output_replaced

@pytest.mark.parametrize(
    "string, tested_th, tested_en",
    [
        ("Hello world", False, False),
        ('สวัสดี \\" ซาวาน', False, False),
        ("สวัสดี \\' ซาวาน", False, False),
        ("สวัสดี \\'\\' ซาวาน", False, False),
        ("สวัสดี ซาวาน &amp; นกกา", False, False),
        ("สวัสดี เเก",  True, False),
        ("สวัสดี กาาา",  True, False),
        ("สวัสดี กำำำำำ",  True, False),
        ("สวัสดี ก่่่่า",  True, False),

    ]
)
def test_NormalizeThaiVowel_test(string, tested_th, tested_en):
    rule = NormalizeThaiVowel()
    output_tested = rule.test(string, lang="th")
    assert tested_th == output_tested

    output_tested = rule.test(string, lang="en")
    assert tested_en == output_tested

@pytest.mark.parametrize(
    "string, replaced_th, replaced_en",
    [
        ("Hello world", "Hello world", "Hello world"),
        ('สวัสดี \\" ซาวาน', 'สวัสดี \\" ซาวาน', 'สวัสดี \\" ซาวาน'),
        ("สวัสดี เเก",  "สวัสดี แก", "สวัสดี เเก"),
        ("สวัสดี กาาา",  "สวัสดี กา", "สวัสดี กาาา"),
        ("สวัสดี กำำำำำ",  "สวัสดี กำ", "สวัสดี กำำำำำ"),
        ("สวัสดี ก่่่า",  "สวัสดี ก่า", "สวัสดี ก่่่า"),
    ]
)
def test_NormalizeThaiVowel_replaced(string, replaced_th, replaced_en):
    rule = NormalizeThaiVowel()
    output_replaced = rule.replace(string, lang="th")
    assert replaced_th == output_replaced

    output_replaced = rule.replace(string, lang="en")
    assert replaced_en == output_replaced

@pytest.mark.parametrize(
    "string, tested",
    [
        ("โคอิสุรุ", False),
        ("Hello world", False),
        ('ก    ', True),
        ('    ', True),
        (' ', True),
        ('_', True),
        ('A', True),
        ('', True),
    ]
)
def test_SentenceLengthLessThanOrEqualOne_test(string, tested):
    rule = SentenceLengthLessThanOrEqualToOne()
    output_tested = rule.test(string, lang="th")
    assert tested == output_tested

    output_tested = rule.test(string, lang="en")
    assert tested == output_tested


@pytest.mark.parametrize(
    "string, tested",
    [
        ("โคอิสุรุ", False),
        ("Hello world", False),
        # b'\x98\xc2'
        ('เธเธธเธเธเธญเธเธฃเธถเนเธเธฅเนเธฒเธงเนเธฒเธเธฑเธเธกเธฒเธเธณเธญเธฐเนเธฃเธเธตเนเธเธตเน', True),
        # b'\xae\xc2'
        ("ฮะฐbbัtั bะฐwlัn', wัnd bัtัn' thะต bะพnะต. Wัnd lัkะต thัั...", True),
        # b'\xb1\xc2'
        ("ฮnัะตัtry hะพwlัn' ะฐt yะพu, yัbbะตrัn' ัtะพrัะตั.", True),
        # b'\xc2\xb7'
        ("เธÍÍÂÙè·ÕèäË¹ËÅèÐ", True),
        # b'\xc2\x8b'
        ("เนเธเธฅเธตเธขเธเธเธดเธเนเธเนเธ", True),
        # b'\xc3\x83'
        ('GarÃ§onหมายถึงเด็ก', True), 
    ]
)
def test_SentenceContainsUnknownSymbols_test(string, tested):
    rule = SentenceContainsUnknownSymbol()
    output_tested = rule.test(string, lang="th")
    assert tested == output_tested

    output_tested = rule.test(string, lang="en")
    assert tested == output_tested



@pytest.mark.parametrize(
    "string, tested_th, tested_en",
    [
        ("โคอิสุรุ", False, False),
        ("โคอิสุรุ", False, False),
        ("โช I got peace like a river I got peace like a river", True, False),
        ("โช Get th", True, False),
        ("โช I got peace like a river in my soul ", True, False),
        ("โช โช", True, False),
        ("Goto the โช", True, False),
    ]
)
def test_ThaiSentenceContainsNoThaiCharactersPattern_test(string, tested_th, tested_en):
    rule = ThaiSentenceContainsNoThaiCharactersPattern()
    output_tested = rule.test(string, lang="th")
    assert tested_th == output_tested

    output_tested = rule.test(string, lang="en")
    assert tested_en == output_tested

@pytest.mark.parametrize(
    "string, tested_th, tested_en",
    [
        ("โคอิสุรุ", False, False),
        ("โคอิสุรุ.", True, False),
        ("สุดท้าย .ฉันยังคงมี", False, False),
        ("Go to the bracket.", True, False),

    ]
)

def test_RemoveFullStopInThaiSentence_test(string, tested_th, tested_en):
    rule = RemoveFullStopInThaiSentence()
    output_tested = rule.test(string, lang="th")
    assert tested_th == output_tested

    output_tested = rule.test(string, lang="en")
    assert tested_en == output_tested


@pytest.mark.parametrize(
    "string, replaced_th, replaced_en",
    [
        ("โคอิสุรุ", "โคอิสุรุ", "โคอิสุรุ"),
        ("โคอิสุรุ.", "โคอิสุรุ", "โคอิสุรุ"),
        ("สุดท้าย .ฉันยังคงมี", "สุดท้าย .ฉันยังคงมี", "สุดท้าย .ฉันยังคงมี"),
        ("Go to the bracket.", "Go to the bracket", "Go to the bracket"),

    ]
)
def test_RemoveFullStopInThaiSentence_replace(string, replaced_th, replaced_en):
    rule = RemoveFullStopInThaiSentence()
    output_replaced = rule.replace(string, lang="th")
    assert replaced_th == output_replaced

    output_replaced = rule.replace(string, lang="en")
    assert replaced_en == output_replaced



@pytest.mark.parametrize(
    "string, tested",
    [
        ("- ", True),
        ("-  ", True),
        (" - ", True),
        (" -", True),
        ("  -", True),
        ("-โคอิสุ", True),
        (" -โคอิสุรุ", True),
        (" - โคอิสุรุ", True),
        (" - Koisuru-", True),
        (" - Koisuru - at-", True),
        ("Koisuru - at", False),
    ]
)
def test_ReplaceDashInSentenceRule_test(string, tested):
    rule = ReplaceDashInSentence()
    output_tested = rule.test(string, lang="th")
    assert tested == output_tested

    output_tested = rule.test(string, lang="en")
    assert tested == output_tested


@pytest.mark.parametrize(
    "string, replaced",
    [
        ("- ", ""),
        ("-  ", ""),
        (" - ", ""),
        (" -", ""),
        ("  -", ""),
        ("-โคอิสุรุ", "โคอิสุรุ"),
        (" -โคอิสุรุ", "โคอิสุรุ"),
        (" - โคอิสุรุ", "โคอิสุรุ"),
        (" - Koisuru at-", "Koisuru at"),
        (" - Koisuru - at-", "Koisuru - at"),
    ]
)
def test_ReplaceDashInSentenceRule_replace(string, replaced):
    rule = ReplaceDashInSentence()
    output_replaced = rule.replace(string, lang="th")
    assert replaced == output_replaced

    output_replaced = rule.replace(string, lang="en")
    assert replaced == output_replaced

@pytest.mark.parametrize(
    "string, tested",
    [
        ("  ", True),
        ("  ", True),
        (" ​ ", True),
        ("  ", True),
        ("  ", True),
        ("  ", True),
        ("  ", True),
        ("  ", True),
        (" โ ", True),
        (" ​ ", True),
        (" ♪ ", True),
    ]
)
def test_RemoveUnwantedSymbols_test(string, tested):
    rule = RemoveUnwantedSymbols()
    output_tested = rule.test(string, lang="th")
    assert tested == output_tested

    output_tested = rule.test(string, lang="en")
    assert tested == output_tested


@pytest.mark.parametrize(
    "string, replaced",
    [
        ("  ", "  "),
        ("  ", "  "),
        (" ​ ", "  "),
        ("  ", "  "),
        ("  ", "  "),
        ("  ", "  "),
        ("  ", "  "),
        ("  ", "  "),
        (" โ ", "  "),
        (" ​ ", "  "),
        (" ♪ ", "  "),
    ]
)
def test_RemoveUnwantedSymbols_replace(string, replaced):
    rule = RemoveUnwantedSymbols()
    output_replaced = rule.replace(string, lang="th")
    assert replaced == output_replaced

    output_replaced = rule.replace(string, lang="en")
    assert replaced == output_replaced


@pytest.mark.parametrize(
    "string, tested",
    [
        ("{\\ cHFFFFFF", True),
        ("{\\cHFFFFFF", True),
        ("{\\\\cHFFFFFF", True),
        ("{\\ cHFFFFFF}", True),
        ("{\\cHFFFFFF}", True),
        ("{\\\\cHFFFFFF}", True),
        ("{\\ cHFFFFFF }", True),
        ("{\\cHFFFFFF }", True),
        ("{\\\\cHFFFFFF }", True),
        ("cHFFFFFF", True),
        ("\\cHFFFFFF", True),
        ("I play {}.",  True),
        ("\ N",  True),
        ("\ NI",  True),
        ("\\\\ i1",  True),
        ("\\",  True),
        ("\\\\",  True),
        ("\\\\\\",  True),
        ("\\\\\\\\",  True),
        ("{}",  True),
        (",8203;", True),
        (",8203;", True),
        ("8203;", True),
        ("#8203;", True),
        ("#8203;", True),
        ("aa 8203", False),
        ('font color = "# 808080 "', True),
        ('font color="# 808080 "', True),
        ('font color="#808080"', True),
        ('font color= "#808080"', True),
    ]
)
def test_RemoveUnwantedPattern_test(string, tested):
    rule = RemoveUnwantedPattern()
    output_tested = rule.test(string, lang="th")
    assert tested == output_tested

    output_tested = rule.test(string, lang="en")
    assert tested == output_tested


@pytest.mark.parametrize(
    "string, replaced",
    [
        ("{\\ cHFFFFFF", ""),
        ("{\\cHFFFFFF", ""),
        ("{\\\\cHFFFFFF", ""),
        ("{\\ cHFFFFFF}", ""),
        ("{\\cHFFFFFF}", ""),
        ("{\\\\cHFFFFFF}", ""),
        ("{\\ cHFFFFFF }", ""),
        ("{\\cHFFFFFF }", ""),
        ("{\\\\cHFFFFFF }", ""),
        ("cHFFFFFF", ""),
        ("\\cHFFFFFF", ""),
        ("I play {}.", "I play ."),
        ("\ N",  ""),
        ("\ NI",  ""),
        ("\\\\ i1",  ""),
        ("\\",  ""),
        ("\\\\",  ""),
        ("\\\\\\",  ""),
        ("\\\\\\\\",  ""),
        ("{}{}.",  "."),
        (",8203;", ""),
        (",8203;", ""),
        ("8203;", ""),
        ("#8203;", ""),
        ("#8203;", ""),
        ("aa 8203", "aa 8203"),
        ('font color = "# 808080 "', ""),
        ('font color="# 808080 "', ""),
        ('font color="#808080"', ""),
        ('font color= "#808080"', ""),


    ]
)
def test_RemoveUnwantedPattern_replace(string, replaced):
    rule = RemoveUnwantedPattern()
    output_replaced = rule.replace(string, lang="th")
    assert replaced == output_replaced

    output_replaced = rule.replace(string, lang="en")
    assert replaced == output_replaced


@pytest.mark.parametrize(
    "string_tuple, tested",
    [
        (("สวสาัดี  a hello world", "a hello world"), True),
        (("นี่มัน 30;", "30;"), True),
        (("โทโมะซัง สบายดีไหม" , " Tomosan how are you "), False),
        (("ฉันจบจาก MIT" , " I am graduated from MIT"), False),

    ]
)
def test_SentencePairFoundRepeatedText_test(string_tuple, tested):
    rule = SentencePairFoundRepeatedText()
    output_tested = rule.test(string_tuple)
    assert tested == output_tested

    output_tested = rule.test(string_tuple)
    assert tested == output_tested
