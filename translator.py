import re
from abc import ABC, abstractmethod
import os
import hazm


class Translator(ABC):

    def __init__(self):
        # replacing non-standard chars with thier equivalent
        self.letter_dict = dict()
        self.letter_dict[u"ۀ"] = u"ه"
        self.letter_dict[u"ة"] = u"ت"
        self.letter_dict[u"ي"] = u"ی"
        self.letter_dict[u"ؤ"] = u"و"
        self.letter_dict[u"إ"] = u"ا"
        self.letter_dict[u"ٹ"] = u"ت"
        self.letter_dict[u"ڈ"] = u"د"
        self.letter_dict[u"ئ"] = u"ی"
        self.letter_dict[u"ﻨ"] = u"ن"
        self.letter_dict[u"ﺠ"] = u"ج"
        self.letter_dict[u"ﻣ"] = u"م"
        self.letter_dict[u"ﷲ"] = u""
        self.letter_dict[u"ﻳ"] = u"ی"
        self.letter_dict[u"ٻ"] = u"ب"
        self.letter_dict[u"ٱ"] = u"ا"
        self.letter_dict[u"ڵ"] = u"ل"
        self.letter_dict[u"ﭘ"] = u"پ"
        self.letter_dict[u"ﻪ"] = u"ه"
        self.letter_dict[u"ﻳ"] = u"ی"
        self.letter_dict[u"ٻ"] = u"ب"
        self.letter_dict[u"ں"] = u"ن"
        self.letter_dict[u"ٶ"] = u"و"
        self.letter_dict[u"ٲ"] = u"ا"
        self.letter_dict[u"ہ"] = u"ه"
        self.letter_dict[u"ﻩ"] = u"ه"
        self.letter_dict[u"ﻩ"] = u"ه"
        self.letter_dict[u"ك"] = u"ک"
        self.letter_dict[u"ﺆ"] = u"و"
        self.letter_dict[u"أ"] = u"ا"
        self.letter_dict[u"ﺪ"] = u"د"
        # this function replace the keys of letter_dict with its values
        self.letter_pattern = re.compile(
            r"(" + "|".join(self.letter_dict.keys()) + r")")
        # replacing all spaces,hyphens,  with white space
        self.space_pattern = re.compile(
            r"[\xad\ufeff\u200e\u200d\u200b\x7f\u202a\u2003\xa0\u206e\u200c\x9d\u200C\u200c\u2005\u2009\u200a\u202f\t\u200c]+")
        # to be removed
        self.deleted_pattern = re.compile(
            r"([^\w\s]|\d|[\|\[]]|\"|'ٍ|[0-9]|¬|[a-zA-Z]|[؛“،,”‘۔’’‘–]|[|\.÷+\:\-\?»\=\{}\*«_…\؟!/ـ]|[۲۹۱۷۸۵۶۴۴۳]|[\\u\\x]|[\(\)]|[۰'ٓ۫'ٔ]|[ٓٔ]|[ًٌٍْﹼ،َُِّ«ٰ»ٖء]|\[]|\[\])")
        self.special_chars = re.compile('|'.join(['²', '³', 'µ', 'À', 'Á', 'Â', 'Ç', 'È', 'É', 'Ê', 'Í', 'Ö', 'Û', 'Ü', 'ß', 'à', 'á', 'â', 'ã', 'ä', 'ç', 'è', 'é', 'ê', 'í', 'î', 'ï', 'ñ', 'ó', 'ô', 'õ', 'ö', 'ø', 'ú', 'ü', 'ý', 'ā', 'č', 'ē', 'ę', 'ğ', 'ī', 'ı', 'ř', 'ş', 'š', 'ū', 'ž', 'Ɛ', 'Α', 'Κ',
                                        'ά', 'ή', 'α', 'β', 'ε', 'ζ', 'η', 'ι', 'κ', 'ν', 'ο', 'π', 'ρ', 'ς', 'τ', 'υ', 'ό', 'Տ', 'ա', 'ե', 'թ', 'ի', 'ն', 'վ', 'ւ', 'ք', 'ḥ', 'ṭ', 'ῥ', '۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹', '٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩', '۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹', ]))
        # for merging multiple consecutive newlines
        self.new_line_pattern = re.compile(r'\n+')
        self.normalizer = hazm.Normalizer()

    @abstractmethod
    def load(self, path: str):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def translate(self, src_txt: str):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self, args):
        pass

    def clean(self, string: str) -> str:
        string = self.space_pattern.sub(" ", string)
        string = self.deleted_pattern.sub("", string)
        string = self.special_chars.sub("", string)
        string = self.letter_pattern.sub(
            lambda x: self.letter_dict[x.group()], string)
        string = self.new_line_pattern.sub(r'\n', string)
        string = self.normalizer.normalize(string)

        return string
