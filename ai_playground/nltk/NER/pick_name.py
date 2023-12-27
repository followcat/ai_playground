import collections

import spacy
import pypinyin
from nerpy import NERModel

from transformers import MBartForConditionalGeneration, MBart50Tokenizer

translate_model_name = 'facebook/mbart-large-50-many-to-many-mmt'
translate_tokenizer = MBart50Tokenizer.from_pretrained(translate_model_name)
translate_model = MBartForConditionalGeneration.from_pretrained(translate_model_name)


def judge_PER(name, tag):
    if tag == 'PER':
        return True
    return False


def judge_ORG(name, tag):
    if tag == 'ORG' and len(name) > 3:
        return True
    return False


def judge_LOC(name, tag):
    if tag == 'LOC' and len(name) > 2:
        return True
    return False


def pick_NER(text, model="shibing624/bert4ner-base-chinese", OPTION=["PER", "ORG", "LOC"]):
    # model = NERModel("bert", "shibing624/bert4ner-base-chinese")
    # model = NERModel("bertspan", "shibing624/bertspan4ner-base-chinese")
    model = NERModel("bert", model)
    nlp = spacy.load('zh_core_web_sm')
    doc = nlp(text)
    texts = [sent.text for sent in doc.sents]
    predictions, raw_outputs, entities = model.predict(texts, split_on_space=False)
    results = collections.defaultdict(list)
    for output in entities:
        for name, tag in output:
            if "PER" in OPTION and judge_PER(name, tag):
                results["PER"].append(
                    (name,
                     ''.join([p.capitalize() for p in pypinyin.lazy_pinyin(name)]),
                     translate_zh_to_en(name))
                )
            if "ORG" in OPTION and judge_ORG(name, tag):
                results["ORG"].append(
                    (name,
                     ''.join([p.capitalize() for p in pypinyin.lazy_pinyin(name)]),
                     translate_zh_to_en(name))
                )
            if "LOC" in OPTION and judge_LOC(name, tag):
                results["LOC"].append(
                    (name,
                     ''.join([p.capitalize() for p in pypinyin.lazy_pinyin(name)]),
                     translate_zh_to_en(name))
                )
    return results


def translate_zh_to_en(text):
    translate_tokenizer.src_lang = "zh_CN"
    encoded_zh = translate_tokenizer(text, return_tensors="pt")
    generated_tokens = translate_model.generate(**encoded_zh, forced_bos_token_id=translate_tokenizer.lang_code_to_id["en_XX"])
    translation = translate_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return translation


if __name__ == '__main__':
    text = """
原告常已英，女，1967年11月23日出生，汉族，住湖北省监利县。
委托代理人夏明、崔亿，均系湖南人和律师事务所律师。
被告柳焕，男，1990年12月28日出生，汉族，住湖南省长沙县。
被告广汽菲亚特克莱斯勒汽车有限公司，住所地：湖南省长沙经济开发区映霞路18号。
法定代表人冯兴亚，董事长。
委托代理人周正祥，男，1981年9月17日出生，汉族，住长沙市芙蓉区。系公司员工。
委托代理人左军，男，1982年2月21日出生，汉族，住湖南省长沙市芙蓉区。系公司员工。
被告中国太平洋财产保险股份有限公司长沙中心支公司，住所地：湖南省长沙市芙蓉区解放西路41号太平洋保险大厦1楼、8-15楼。
负责人赵子安，总经理。
委托代理人赵尚军，湖南路衡律师事务所律师。
被告李文安，男，1986年9月25日出生，汉族，住湖南省长沙县。
被告阳光财产保险股份有限公司湖南省分公司，住所地：湖南省长沙市天心区友谊路413号运成大厦18、19楼。
负责人胡湘泽，总经理。
委托代理人刘滨，男，1960年12月31日出生，汉族，住湖南省长沙市天兴区。
"""
    print(translate_zh_to_en(text))
    results = pick_NER(text)
    for key in results:
        print(key)
        for each in results[key]:
            print(each)


