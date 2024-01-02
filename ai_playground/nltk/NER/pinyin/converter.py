import re

from NER.pinyin.collection import Collection
from NER.pinyin.data.loader import chars
from NER.pinyin.data.loader import surnames
from NER.pinyin.data.loader import words


class Converter:
    TONE_STYLE_SYMBOL = 'symbol'
    TONE_STYLE_NUMBER = 'number'
    TONE_STYLE_NONE = 'none'
    SEGMENTS_COUNT = 10

    def __init__(self):
        self._polyphonic = False
        self._polyphonic_as_list = False
        self.as_surname = False
        self.no_words = False
        self.cleanup = True
        self.yu_to = 'v'
        self.tone_style = self.TONE_STYLE_SYMBOL
        self.regexps = {
            'number': '0-9',
            'alphabet': 'a-zA-Z',
            'hans': '[\u3007\u2E80-\u2FFF\u3100-\u312F\u31A0-\u31EF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]',
            'punctuation': r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]',
        }
        # 合并自定义正则表达式
        self.regexps.update({
            'separator': r'\s',
            'mark': r'',
            'tab': r'\t',
        })

    @classmethod
    def make(cls):
        return cls()

    def polyphonic(self, as_list=False):
        self._polyphonic = True
        self._polyphonic_as_list = as_list
        return self

    def surname(self):
        self.as_surname = True
        return self

    def no_words(self):
        self.no_words = True
        return self

    def no_cleanup(self):
        self.cleanup = False
        return self

    def only_hans(self):
        self.regexps['hans'] = self.regexps['hans']
        return self.no_alpha().no_number().no_punctuation()

    def no_alpha(self):
        self.regexps.pop('alphabet', None)
        return self

    def no_number(self):
        self.regexps.pop('number', None)
        return self

    def no_punctuation(self):
        self.regexps.pop('punctuation', None)
        return self

    def with_tone_style(self, tone_style):
        self.tone_style = tone_style
        return self

    def no_tone(self):
        self.tone_style = self.TONE_STYLE_NONE
        return self

    def use_number_tone(self):
        self.tone_style = self.TONE_STYLE_NUMBER
        return self

    def yu_to_yu(self):
        self.yu_to = 'yu'
        return self

    def yu_to_v(self):
        self.yu_to = 'v'
        return self

    def yu_to_u(self):
        self.yu_to = 'u'
        return self

    def when(self, condition, callback):
        if condition:
            callback(self)
        return self

    def convert(self, string, before_split=None):
        # 分离数字和汉字
        string = re.sub(r'[a-z0-9_-]+', lambda m: "\t" + m.group(0), string, flags=re.I)

        # 过滤掉不保留的字符
        if self.cleanup:
            pattern = '[^0-9a-zA-Z\u3007\u2E80-\u2FFF\u3100-\u312F\u31A0-\u31EF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]'
            string = re.sub(pattern, '', string)

        # 处理多音字
        if self._polyphonic:
            return self.convert_as_chars(string, True)

        if self.no_words:
            return self.convert_as_chars(string)

        # 替换姓氏
        if self.as_surname:
            string = self.convert_surname(string)

        # 替换词语
        for word in words:
            string = self.replace_words(string, word)

        # 分割字符串
        if before_split:
            string = before_split(string)
        return self.split(string)

    def convert_as_chars(self, string, polyphonic=False):
        items = []
        for char in string:
            if char in chars:
                pinyin_list = chars[char]
                if polyphonic:
                    formatted_pinyin = [self.format_tone(pinyin, self.tone_style) for pinyin in pinyin_list]
                    if self._polyphonic_as_list:
                        items.append({char: formatted_pinyin})
                    else:
                        items.extend(formatted_pinyin)
                else:
                    items.append(self.format_tone(pinyin_list[0], self.tone_style))
        return Collection(items)

    def convert_surname(self, name):
        for surname, pinyin in surnames.items():
            if name.startswith(surname):
                return pinyin + name[len(surname):]
        return name

    def replace_words(self, string, words_dict):
        for word, pinyin in words_dict.items():
            string = string.replace(word, pinyin)
        return string

    def split(self, string):
        items = re.split(r'\s+', string)
        return Collection([self.format_tone(item, self.tone_style) for item in items if item])

    def format_tone(self, pinyin, style):
        # 这里需要根据具体的拼音格式化逻辑实现
        # 省略了具体的拼音格式化逻辑
        return pinyin

    def format_tone(self, pinyin, style):
        if style == self.TONE_STYLE_SYMBOL:
            return pinyin

        replacements = {
            'ɑ': ('a', 5), 'ü': ('v', 5),
            'üē': ('ue', 1), 'üé': ('ue', 2), 'üě': ('ue', 3), 'üè': ('ue', 4),
            'ā': ('a', 1), 'ē': ('e', 1), 'ī': ('i', 1), 'ō': ('o', 1), 'ū': ('u', 1), 'ǖ': ('v', 1),
            'á': ('a', 2), 'é': ('e', 2), 'í': ('i', 2), 'ó': ('o', 2), 'ú': ('u', 2), 'ǘ': ('v', 2),
            'ǎ': ('a', 3), 'ě': ('e', 3), 'ǐ': ('i', 3), 'ǒ': ('o', 3), 'ǔ': ('u', 3), 'ǚ': ('v', 3),
            'à': ('a', 4), 'è': ('e', 4), 'ì': ('i', 4), 'ò': ('o', 4), 'ù': ('u', 4), 'ǜ': ('v', 4),
        }

        for unicode, replacement in replacements.items():
            if unicode in pinyin:
                umlaut, tone_number = replacement
                if self.yu_to != 'v' and umlaut == 'v':
                    umlaut = self.yu_to
                pinyin = pinyin.replace(unicode, umlaut)
                if style == self.TONE_STYLE_NUMBER:
                    pinyin += str(tone_number)

        return pinyin

