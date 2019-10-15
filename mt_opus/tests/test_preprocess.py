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
    ReplaceAsteriskInSentence,
    SentenceContainsOnlyAsterisk,
    RemoveColonInSentence,
    RemoveSemiColonInSentence,
    FormatTime,
    ThaiSentenceContainsUnwantedPattern,
    SentencePairTokenLengthsDifferGreaterThreshold,
    EnglishSentenceContainsUnwantedPattern,
    SentencePairUSESimilarityLessThanThreashold
)

@pytest.mark.parametrize(
     "string_pair, tested",
    [   
        # misspelled
        (("ฉันสงสัยว่าเ? ด็ก ๆ ",
          "I wonder if the children are..."),
         True),
        (("ฉันเดิน",
          "I walk"),
         False),
        (("ฉันซื้อแมว", "I bought a cat."),
         False
        )
    ]
)
@pytest.mark.skip(reason="not work currently on mac osx")
def test_SentencePairUSESimilarityLessThanThreashold_test(string_pair, tested):
    rule = SentencePairUSESimilarityLessThanThreashold(threshold=0.4)
    output_tested = rule.test(string_pair)
    assert tested == output_tested

    output_tested = rule.test(string_pair)
    assert tested == output_tested


@pytest.mark.parametrize(
     "string_pair, tested",
    [
        (("เขาจะต้องลำบากแน่ๆ เวลาเดิน",
          "He will have the embarrassment of walking the entire floor."),
          False),
        (("ฉันเดิน",
          "I walk"),
         False),
        (("ไม่กัดวัน", "Not a bit for days."),
         False
        ),
         (("ไม่", "No. Not a bit for days.,"),
         True
        )
    ]
)
def test_SentencePairTokenLengthsDifferGreaterThreshold_test(string_pair, tested):
    rule = SentencePairTokenLengthsDifferGreaterThreshold(threshold=0.85)
    output_tested = rule.test(string_pair)
    assert tested == output_tested

    output_tested = rule.test(string_pair)
    assert tested == output_tested


@pytest.mark.parametrize(
     "string, tested_th, tested_en",
    [
        ("เพลง", True, False),
        ("*เพลง*", True, False),
        ("*เพลง ไทย", False, False),
        ("คำบรรยายโดย", True, False),
    ]
)
def test_ThaiSentenceContainsUnwantedPattern_test(string, tested_th, tested_en):
    rule = ThaiSentenceContainsUnwantedPattern()
    output_tested = rule.test(string, lang="th")
    assert tested_th == output_tested

    output_tested = rule.test(string, lang="en")
    assert tested_en == output_tested

@pytest.mark.parametrize(
     "string, tested_th, tested_en",
    [
        ("Subtitles conformed by SOFTITLER", False, True),
        ("Yemanja be praised. Visiontext Subtitles", False, True),
        ("Subtitle By Trasporter", False, True),
        ("Visiontext subtitles by Susan Voas", False, True)
    ]
)
def test_EnglishSentenceContainsUnwantedPattern_test(string, tested_th, tested_en):
    rule = EnglishSentenceContainsUnwantedPattern()
    output_tested = rule.test(string, lang="th")
    assert tested_th == output_tested

    output_tested = rule.test(string, lang="en")
    assert tested_en == output_tested



@pytest.mark.parametrize(
     "string, tested",
    [
        ("Hello world:", True),
        ("12:29", False),
    ]
)
def test_RemoveColonInSentence_test(string, tested):
    rule = RemoveColonInSentence()
    output_tested = rule.test(string, lang="th")
    assert tested == output_tested

    output_tested = rule.test(string, lang="en")
    assert tested == output_tested

@pytest.mark.parametrize(
     "string, replaced",
    [
        ("Hello world:", "Hello world"),
        ("12:29", "12:29"),
    ]
)
def test_RemoveColonInSentence_replace(string, replaced):
    rule = RemoveColonInSentence()
    output_replaced = rule.replace(string, lang="th")
    assert replaced == output_replaced

    output_replaced = rule.replace(string, lang="en")
    assert replaced == output_replaced


@pytest.mark.parametrize(
     "string, tested",
    [
        ("Hello world;", True),
        ("Hello world ;", True),
        ("12;29", False),
    ]
)
def test_RemoveSemiColonInSentence_test(string, tested):
    rule = RemoveSemiColonInSentence()
    output_tested = rule.test(string, lang="th")
    assert tested == output_tested

    output_tested = rule.test(string, lang="en")
    assert tested == output_tested

@pytest.mark.parametrize(
     "string, replaced",
    [
        ("Hello world;", "Hello world"),
        ("Hello world ;", "Hello world"),
        ("12:29", "12:29"),
    ]
)
def test_RemoveSemiColonInSentence_replace(string, replaced):
    rule = RemoveSemiColonInSentence()
    output_replaced = rule.replace(string, lang="th")
    assert replaced == output_replaced

    output_replaced = rule.replace(string, lang="en")
    assert replaced == output_replaced



@pytest.mark.parametrize(
     "string, tested",
    [
        ("12:29", False),
    ]
)
def test_FormatTime_test(string, tested):
    rule = FormatTime()
    output_tested = rule.test(string, lang="th")
    assert tested == output_tested

    output_tested = rule.test(string, lang="en")
    assert tested == output_tested

@pytest.mark.parametrize(
     "string, replaced",
    [
        ("12:29", "12:29"),
        ("12: 29", "12:29"),
        ("เวลา09: 00", "เวลา09:00"),
        ("เวลา 09: 00 น.", "เวลา 09:00 น."),
    ]
)
def test_FormatTime_replace(string, replaced):
    rule = FormatTime()
    output_replaced = rule.replace(string, lang="th")
    assert replaced == output_replaced

    output_replaced = rule.replace(string, lang="en")
    assert replaced == output_replaced




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
        ("สวัสดี \\\\'\\\\' ซาวาน", "สวัสดี '' ซาวาน"),
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
        ('', True),
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
        ("Koisuru - at", True),
        ("แหม-", True),
        ("ไม่ดีกว่าค่ะ-", True),
        ("ทำไม--เธอทำอย่างนั้นทำไม?", True),
        ("ทำไม---เธอทำอย่างนั้นทำไม?", True),
        ("This building - - I swear to god.", True),
        ("This building - - - I swear to god.", True),
        ("Noooo---Let me out...!", True),
        ("นั่นคือที่เค้าว่ากันนะ ---CSI ปีที่7 ตอนที่21--- ---\"จุดจบของแฮปปี้\"", True),
        ("I'm Sorry.---Pete", True),
        ("You, young lady, --- - I'm a mother.", True),
        ("You want to live with that b---- ?", True),
        ("Well, w-- uh, I", True),
        ("Well, w-- uh, I", True),
        ("Why, it's-- it's Pinoke.", True),
        ("But I'm going-- - Straight to the top.", True),
        ("P-I-N- - Eh, uh-- U-O", True),
        ("What the--?", True),
        ("อะไรหน่ะ- -?", True),
        ("You--!", True),
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
        (" - Koisuru - at-", "Koisuru at"),
        ("แหม-", "แหม"),
        ("ไม่ดีกว่าค่ะ-", "ไม่ดีกว่าค่ะ"),
        ("ดูรายชื่อโทรออก--- ส่วนใหญ่ถูกลบไปแล้ว มี 2 เบอร์เป็นของทหารในซานดิเอโก้", "ดูรายชื่อโทรออก ส่วนใหญ่ถูกลบไปแล้ว มี 2 เบอร์เป็นของทหารในซานดิเอโก้"),
        ("Ran the calls-- Mainly take-out, couple to some jarheads in San Diego.", "Ran the calls Mainly take-out, couple to some jarheads in San Diego."),
        ("ทำไม--เธอทำอย่างนั้นทำไม?", "ทำไม เธอทำอย่างนั้นทำไม?"),
        ("ทำไม---เธอทำอย่างนั้นทำไม?", "ทำไม เธอทำอย่างนั้นทำไม?"),
        ("ไม่ใช่ ฉันไม่ต้อง---- ไม่ คือฉัน", "ไม่ใช่ ฉันไม่ต้อง ไม่ คือฉัน"),
        ("This building - - I swear to god.", "This building I swear to god."),
        ("This building - - - I swear to god.", "This building I swear to god."),
        ("Noooo---Let me out...!", "Noooo Let me out...!"),
        ("นั่นคือที่เค้าว่ากันนะ ---CSI ปีที่7 ตอนที่21--- ---\"จุดจบของแฮปปี้\"", "นั่นคือที่เค้าว่ากันนะ CSI ปีที่7 ตอนที่21 \"จุดจบของแฮปปี้\""),
        ("I'm Sorry.---Pete", "I'm Sorry. Pete"),
        ("You, young lady, --- - I'm a mother.", "You, young lady, I'm a mother."),
        ("it's nooo prob--- tell me!", "it's nooo prob tell me!"),
        ("You want to live with that b---- ?", "You want to live with that b ?"),
        ("Well, w-- uh, I", "Well, w uh, I"),
        ("Why, it's-- it's Pinoke.", "Why, it's it's Pinoke."),
        ("But I'm going-- - Straight to the top.","But I'm going Straight to the top."),
        ("P-I-N- - Eh, uh-- U-O", "P-I-N Eh, uh U-O"),
        ("\"c--deny everything, and d--all 3.\"", "\"c deny everything, and d all 3.\""),
        ("What the--?", "What the ?"),
        ("อะไรหน่ะ- -?", "อะไรหน่ะ ?"),
        ("You--!", "You !"),
        ("อะ--", "อะ"),
        ("Bones ปีที่ 7 ตอนที่ 8--The Bump in the Road ออกอากาศวันที่ 9 เมษายน 2555",
         "Bones ปีที่ 7 ตอนที่ 8 The Bump in the Road ออกอากาศวันที่ 9 เมษายน 2555"),
        ("น้องเมียอาจมีส่วนเกี่ยวข้องกับ การตายของน้องชายตัวเอง เราพบเงินสดในล็อคเกอร์ของคุึณ ที่ควีนส์---",
         "น้องเมียอาจมีส่วนเกี่ยวข้องกับ การตายของน้องชายตัวเอง เราพบเงินสดในล็อคเกอร์ของคุึณ ที่ควีนส์"),
        ("และจากที่ฉันได้ยินในอุโมงค์--/เธอบอกว่าเธอไม่ได้เห็นอะไรหนิ", "และจากที่ฉันได้ยินในอุโมงค์ เธอบอกว่าเธอไม่ได้เห็นอะไรหนิ"),
        ("And from what I heard in that tunnel -You said you didn't see anything.", "And from what I heard in that tunnel You said you didn't see anything."),
        ("เบื้องต้นนะ--22 คดี", "เบื้องต้นนะ 22 คดี"),
        ("1--ยกเลิกโทษประหาร", "1 ยกเลิกโทษประหาร"),
        

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
        (("นี่มัน 30;", ""), True),
        (("โทโมะซัง สบายดีไหม" , " Tomosan how are you "), False),
        (("ฉันจบจาก MIT" , " I am graduated from MIT"), False),
        (("Ask my sisters - แม่!", "Ask my sisters "), True)

    ]
)
def test_SentencePairFoundRepeatedText_test(string_tuple, tested):
    rule = SentencePairFoundRepeatedText()
    output_tested = rule.test(string_tuple)
    assert tested == output_tested

    output_tested = rule.test(string_tuple)
    assert tested == output_tested




@pytest.mark.parametrize(
    "string, tested",
    [
        ("***The joys of surgeondom.", True),
        ("* she looked at me with big brown eyes * [ knock on door ]", True),
        ("We got a *bollo* on Little Chino.", True),
        ("Frank lundy? *******", True),
        ("* and said, \"you ain\'t seen nothing yet\" *", True),
        ("* * you want it, you got it, uh * * you want it, baby, you got it * * your best friend harry has a brother larry * * in five days from now, he's gonna marry *", True),
        ("* Lord *", True)
    ]
)
def test_ReplaceAsteriskInSentencee_test(string, tested):
    rule = ReplaceAsteriskInSentence()
    output_tested = rule.test(string, lang="th")
    assert tested == output_tested

    output_tested = rule.test(string, lang="en")
    assert tested == output_tested


@pytest.mark.parametrize(
    "string, replaced",
    [
        ("***The joys of surgeondom.", "The joys of surgeondom."),
        ("* she looked at me with big brown eyes * [ knock on door ]", "she looked at me with big brown eyes [ knock on door ]"),
        ("We got a *bollo* on Little Chino.", "We got a bollo on Little Chino."),
        ("Frank lundy? *******", "Frank lundy?"),
        ("* and said, \"you ain\'t seen nothing yet\" *", "and said, \"you ain\'t seen nothing yet\""),
        ("* * you want it, you got it, uh * * you want it, baby, you got it * * your best friend harry has a brother larry * * in five days from now, he's gonna marry *",
         "you want it, you got it, uh you want it, baby, you got it your best friend harry has a brother larry in five days from now, he's gonna marry"),
        ("* Lord *", "Lord"),
        ("* พระองค์ *", "พระองค์"),
        ("* *deleted*", "deleted"),
    ]
)
def test_ReplaceAsteriskInSentence_replace(string, replaced):
    rule = ReplaceAsteriskInSentence()
    output_replaced = rule.replace(string, lang="th")
    assert replaced == output_replaced

    output_replaced = rule.replace(string, lang="en")
    assert replaced == output_replaced


@pytest.mark.parametrize(
    "string, tested",
    [
        ("***", True),
        (" *******", True),
        ("******* ", True),
        ("  ******* ", True),
        ("  ******* a cat", False),
    ]
)
def test_SentenceContainsOnlyAsterisk_test(string, tested):
    rule = SentenceContainsOnlyAsterisk()
    output_tested = rule.test(string, lang="th")
    assert tested == output_tested

    output_tested = rule.test(string, lang="en")
    assert tested == output_tested

