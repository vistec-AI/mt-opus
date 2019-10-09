# coding: utf-8
from __future__ import unicode_literals

import pytest
from mt_opus.preprocess import (
    UnescapeString,
    RemoveHashtagInSentenceRule,
    NormalizeThaiVowel,
    SentenceLengthLessThanOrEqualToOne,
    SentenceContainsUnknownSymbol,
    SentenceContainsAdSymbol,
    ThaiSentenceContainsNoThaiCharacters,
    ThaiSentenceContainsNoThaiCharactersPattern
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
def test_RemoveHashtagInSentenceRule_test(string, tested):
    rule = RemoveHashtagInSentenceRule()
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

def test_RemoveHashtagInSentenceRule_replace(string, replaced):
    rule = RemoveHashtagInSentenceRule()
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
        ("โช Get th", True, False),
        ("โคอิสุรุ", True, False),
        ("โช โช", True, False),
        ("โคอิสุรุ", True, False),

    ]
)
def test_ThaiSentenceContainsNoThaiCharactersPattern_test(string, tested_th, tested_en):
    rule = ThaiSentenceContainsNoThaiCharactersPattern()
    output_tested = rule.test(string, lang="th")
    assert tested_th == output_tested

    output_tested = rule.test(string, lang="en")
    assert tested_en == output_tested
