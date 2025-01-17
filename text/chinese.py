import os
import re

import cn2an
from pypinyin import lazy_pinyin, Style

from text.symbols import punctuation
from text.tone_sandhi import ToneSandhi
#from tn.chinese.normalizer import Normalizer
from text.zh_normalization.text_normlization import TextNormalizer
from text.english import g2p_w as g2p_en_w

current_file_path = os.path.dirname(__file__)
pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()
}
an2cn_normalizer = TextNormalizer()

from transformers import AutoTokenizer
LOCAL_PATH = "/asrfs/users/wd007/asr/tools/src/opensource/bert-vits2-dev/bert/chinese-roberta-wwm-ext-large"
tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)

EN_WORD_TG = '▁'
#import jieba
#jieba.add_word(EN_WORD_TG)
import jieba.posseg as psg


rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}

tone_modifier = ToneSandhi()

ENCHARS = 'abcdefghijklmnopqrstuvwxyz0123456789'


'''
def _clean_space(text):
  """"
  处理多余的空格
  """
  #match_regex = re.compile(u'[\u4e00-\u9fa5。\.\',，:：《》、\(\)（）]{1} +(?<![a-zA-Z])|\d+ +| +\d+|[a-z A-Z]+')
  match_regex = re.compile(u'[\u4e00-\u9fa5。\.\',，:：《》、\(\)（）' + "".join(punctuation) +  ']{1} +(?<![a-zA-Z])|\d+ +| +\d+|[a-z A-Z]+')
  should_replace_list = match_regex.findall(text)
  order_replace_list = sorted(should_replace_list,key=lambda i:len(i),reverse=True)
  for i in order_replace_list:
    if i == u' ':
      continue
    new_i = i.strip()
    text = text.replace(i,new_i)
  return text
'''


def _clean_space(text):
    """"
    处理多余的空格
    """

    clean_text = ''
    enden = False
    strs = text.split()
    for ss in strs:
        sten = True if ss[0].lower() in ENCHARS else False
        clean_text += ' '+ss if enden and sten else ss
        enden = True if ss[-1].lower() in ENCHARS else False
    return clean_text
        

def replace_punctuation(text):
    text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    #replaced_text = re.sub(r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text)

    ## 保留中英文和指定标点符号
    replaced_text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z " + "".join(punctuation) + r"]+", "", replaced_text)
    ## 去除多余空格，保留英文单词之间空格
    replaced_text = _clean_space(replaced_text)

    return replaced_text


def g2p(text):
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    phones, tones, word2ph = _g2p(sentences)
    #print(phones, tones, word2ph)
    assert sum(word2ph) == len(phones)
    #assert len(word2ph) == len(text)  # Sometimes it will crash,you can add a try-catch.
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph


def _get_initials_finals(word):
    initials = []
    finals = []
    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    orig_finals = lazy_pinyin(
        word, neutral_tone_with_five=True, style=Style.FINALS_TONE3
    )
    for c, v in zip(orig_initials, orig_finals):
        initials.append(c)
        finals.append(v)
    return initials, finals


def _is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def _is_all_punc(strs):
    punc = "".join(punctuation)
    for c in strs:
        if not c in punc:
            return False
    return True


def _split_cn_en(str):
    english = 'abcdefghijklmnopqrstuvwxyz0123456789'
    output = []
    buffer = ''
    for s in str:
        if s in english or s in english.upper(): 
            buffer += s
        else:
            if buffer: output.append(buffer)
            buffer = ''
            s = s.strip()
            if s: output.append(s)
    if buffer: output.append(buffer)
    return output


def _replace_enwords(str):
    english = ENCHARS
    words = []
    word = ''
    oss = ''
    for s in str:
        if s in english or s in english.upper(): 
            word += s
        else:
            if word: 
                words.append(word)
                #oss = oss.strip() + ' #'
                #oss = oss.strip() + ' ▁'
                oss += EN_WORD_TG
            word = ''
            #s = s.strip()
            oss += s
    if word: 
        words.append(word)
        oss += EN_WORD_TG
    return oss, words


def _split_enwords(seg_cut):
    segs = []
    for seg in seg_cut:
        if EN_WORD_TG not in seg[0]:
            segs.append(seg)
            continue
        lsegs = re.split('(▁)', seg[0])
        for s in lsegs:
            if s != '':
                segs.append([s, seg[1]])
    return segs


def split_cn_en(seg_cut):
    segs = []
    for seg in seg_cut:
        if _is_all_chinese(seg[0]) or seg[0].encode('utf-8').isalpha() or _is_all_punc(seg[0]): 
            segs.append(seg)
            continue
        lsegs = _split_cn_en(seg[0])
        for s in lsegs:
            segs.append([s, seg[1]])
    return segs
         
        
def _g2p(segments):
    phones_list = []
    tones_list = []
    word2ph = []
    for seg in segments:
        # Replace all English words in the sentence
        #seg = re.sub("[a-zA-Z]+", "", seg)
        seg, enwords = _replace_enwords(seg)
        #print(seg, enwords)
        seg_cut = psg.lcut(seg, use_paddle=True)
        #print(seg_cut)
        initials = []
        finals = []
        seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)
        #print(seg_cut)
        seg_cut = _split_enwords(seg_cut)
        #print(seg_cut)
        k = 0
        for word, pos in seg_cut:
            #print(word, pos)
            if word == " ":
                continue
            if word == EN_WORD_TG:
                tokens = tokenizer.tokenize(enwords[k])
                phones_en, tones_en, word2ph_en = g2p_en_w(enwords[k], tokens)
                phones_list += phones_en
                tones_list += tones_en
                word2ph += word2ph_en
                k += 1
                continue

            sub_initials, sub_finals = _get_initials_finals(word)
            sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
            #initials.append(sub_initials)
            #finals.append(sub_finals)

            # assert len(sub_initials) == len(sub_finals) == len(word)
            #initials = sum(initials, [])
            #finals = sum(finals, [])
            initials = sub_initials
            finals = sub_finals
            #
            for c, v in zip(initials, finals):
                raw_pinyin = c + v
                # NOTE: post process for pypinyin outputs
                # we discriminate i, ii and iii
                if c == v:
                    assert c in punctuation
                    phone = [c]
                    tone = "0"
                    word2ph.append(1)
                else:
                    v_without_tone = v[:-1]
                    tone = v[-1]
 
                    pinyin = c + v_without_tone
                    assert tone in "12345"
 
                    if c:
                        # 多音节
                        v_rep_map = {
                            "uei": "ui",
                            "iou": "iu",
                            "uen": "un",
                        }
                        if v_without_tone in v_rep_map.keys():
                            pinyin = c + v_rep_map[v_without_tone]
                    else:
                        # 单音节
                        pinyin_rep_map = {
                            "ing": "ying",
                            "i": "yi",
                            "in": "yin",
                            "u": "wu",
                        }
                        if pinyin in pinyin_rep_map.keys():
                            pinyin = pinyin_rep_map[pinyin]
                        else:
                            single_rep_map = {
                                "v": "yu",
                                "e": "e",
                                "i": "y",
                                "u": "w",
                            }
                            if pinyin[0] in single_rep_map.keys():
                                pinyin = single_rep_map[pinyin[0]] + pinyin[1:]
 
                    assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)
                    phone = pinyin_to_symbol_map[pinyin].split(" ")
                    word2ph.append(len(phone))

                #print(word2ph)
                phones_list += phone
                tones_list += [int(tone)] * len(phone)
    return phones_list, tones_list, word2ph


def text_normalize(text):
    #numbers = re.findall(r"\d+(?:\.?\d+)?", text)
    #for number in numbers:
    #    text = text.replace(number, cn2an.an2cn(number), 1)
    # Chinese Numerals To Arabic Numerals
    #text = cn2an.transform(text, "an2cn")
    text = an2cn_normalizer.normalize_sentence(text)
    text = replace_punctuation(text)
    return text


def get_bert_feature(text, word2ph):
    from text import chinese_bert

    return chinese_bert.get_bert_feature(text, word2ph)


if __name__ == "__main__":
    from text.chinese_bert import get_bert_feature

    text = "可我 一问 ，这 玩意儿 并不 靠谱 。"
    text = "啊！但是《原神》是由,米哈\游自主，  [研发]的一款全.新开放世界.冒险游戏"
    text = "瓦拉德 怀疑 此案是 “内鬼 ”所为 "
    text = "其中 ，福州 居于 榜首 。"
    text = "此次 重庆 打黑 审判 ，也已 进入 “扫尾” 阶段 。"
    text = "G P 是吧U显卡 RTX GPU 4080, 啊！但是《原神》是由,米哈\游自主，  … 猪头- !?[研发]的一款全.新开放世界.冒险游戏"
    text = "我是 善良 活泼 、好奇心 旺盛的 B型血 "
    text = "扭一扭,舔一舔,泡一泡."
    text = "富士通推出以人为本的aizinrai系统."
    text = "想想口水又来了,可以保留个 VIP 卡,可以打折哦!"
    text = "服务员总体比较松散,butter喊了几次都没给我拿来."
    text = "富士通推出以人为本的aizinrai butter VIP系统."
    text = "富士通推出以人为本aizinrai的aizinrai butter VIP系统."
    text = "大厅里 有 弹钢琴 ，当时 好像 唱的是 yesterday  once more 。…-"
    text = "一个月期涨零点九一 BP 报百分之四点零三六一。"
    text = "你真是太慷慨了.我觉得这样is ok"
    text = "头发 齐齐地 挂到 耳根 ，走去时 旗袍 在 腰上 一皱 一皱 。"
    text = "一大片一大片,就像潮水从我们身上涌过去."
    text = "TA TA T"
    text = "气球给你们，别抢我 switch"
    text = "剩了 一些 茶卤儿 ，留着 过年吧 。年夜饭的 剩菜 干一干 可香啦 。"
    print(text)
    text = text_normalize(text)
    print(text)
    phones, tones, word2ph = g2p(text)
    print(phones, tones, word2ph)
    bert = get_bert_feature(text, word2ph)
    print(bert.shape)

    #print(phones, tones, word2ph)
    #print(phones, tones, word2ph, bert.shape)


# # 示例用法
# text = "这是一个示例文本：,你好！这是一个测试...."
# print(g2p_paddle(text))  # 输出: 这是一个示例文本你好这是一个测试
