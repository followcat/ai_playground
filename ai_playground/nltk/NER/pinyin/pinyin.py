from converter import Converter
from collection import Collection

class Pinyin:
    @staticmethod
    def name(name, tone_style=Converter.TONE_STYLE_SYMBOL):
        return Converter.make().surname().with_tone_style(tone_style).convert(name)

    @staticmethod
    def passport_name(name, tone_style=Converter.TONE_STYLE_NONE):
        return Converter.make().surname().yu_to_yu().with_tone_style(tone_style).convert(name)

    @staticmethod
    def phrase(string, tone_style=Converter.TONE_STYLE_SYMBOL):
        return Converter.make().no_punctuation().with_tone_style(tone_style).convert(string)

    @staticmethod
    def sentence(string, tone_style=Converter.TONE_STYLE_SYMBOL):
        return Converter.make().with_tone_style(tone_style).convert(string)

    @staticmethod
    def full_sentence(string, tone_style=Converter.TONE_STYLE_SYMBOL):
        return Converter.make().no_cleanup().with_tone_style(tone_style).convert(string)

    @staticmethod
    def polyphones(string, tone_style=Converter.TONE_STYLE_SYMBOL, as_list=False):
        return Converter.make().polyphonic(as_list).with_tone_style(tone_style).convert(string)

    @staticmethod
    def polyphones_as_array(string, tone_style=Converter.TONE_STYLE_SYMBOL):
        return Pinyin.polyphones(string, tone_style, True)

    @staticmethod
    def chars(string, tone_style=Converter.TONE_STYLE_SYMBOL):
        return Converter.make().only_hans().no_words().with_tone_style(tone_style).convert(string)

    @staticmethod
    def permalink(string, delimiter='-'):
        if delimiter not in ['_', '-', '.', '']:
            raise ValueError("Delimiter must be one of: '_', '-', '.', ''.")
        return Converter.make().no_punctuation().no_tone().convert(string).join(delimiter)

    @staticmethod
    def name_abbr(string):
        return Pinyin.abbr(string, True)

    @staticmethod
    def abbr(string, as_name=False):
        converter = Converter.make().no_tone().no_punctuation()
        if as_name:
            converter = converter.surname()
        collection = converter.convert(string)
        return collection.map(lambda pinyin: pinyin[0] if len(pinyin) > 1 else pinyin)

    @staticmethod
    def __getattr__(name):
        def method(*args, **kwargs):
            converter = Converter.make()
            if hasattr(converter, name):
                return getattr(converter, name)(*args, **kwargs)
            raise AttributeError(f"Method {name} does not exist.")
        return method

