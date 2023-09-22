import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import sys

JP_BERT="/asrfs/users/wd007/asr/tools/src/opensource/bert-vits2-dev/bert/bert-base-japanese-v3/"
tokenizer = AutoTokenizer.from_pretrained(JP_BERT)
jp_bert_model = None


def get_bert_feature(text, word2ph, device=None):
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"

    global jp_bert_model
    if jp_bert_model is None:
        jp_bert_model = AutoModelForMaskedLM.from_pretrained(ZH_BERT).to(device)

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = jp_bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
    assert inputs["input_ids"].shape[-1] == len(word2ph)
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T
