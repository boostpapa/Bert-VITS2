
import argparse
import os
import sys
import time

import numpy as np
from scipy.io import wavfile

import torch
import argparse
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence, get_bert
from text.cleaner import clean_text

net_g = None
def get_args():
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--checkpoint', required=True, default="./logs/as/G_8000.pth", help='checkpoint')
    parser.add_argument('--config', required=True, default="./configs/config.json", help='config file')
    parser.add_argument('--outdir', required=True, help='ouput directory')
    parser.add_argument('--test_file', required=True, help='test file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this local rank, -1 for cpu')
    args = parser.parse_args()
    return args


def get_text(hps, text, language_str, device='cuda'):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.add_word_blank:
        phone = commons.intersperse_word(phone, word2ph, 0)
        tone = commons.intersperse_word(tone, word2ph, 0)
        language = commons.intersperse_word(language, word2ph, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] + 1
        word2ph[0] += 1
    elif hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert_ori = get_bert(norm_text, word2ph, language_str, device)
    del word2ph
    assert bert_ori.shape[-1] == len(phone), phone

    bert = torch.zeros(1024, len(phone))
    ja_bert = torch.zeros(1024, len(phone))
    en_bert = torch.zeros(1024, len(phone))
    if language_str == "ZH":
        bert = bert_ori
    elif language_str == "JA":
        ja_bert = bert_ori
    elif language_str == "EN":
        en_bert = bert_ori

    assert bert_ori.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, en_bert, phone, tone, language


def infer(hps, text, sdp_ratio, noise_scale,
          noise_scale_w, length_scale, sid, language, device='cuda'):
    global net_g
    bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(hps, text, language, device)
    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        en_bert = en_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        del phones
        speakers = torch.LongTensor([sid]).to(device)
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                ja_bert,
                en_bert,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers
        return audio


def tts_fn(hps, text, sid, sdp_ratio, noise_scale,
           noise_scale_w, length_scale, language, device='cuda'):
    with torch.no_grad():
        audio = infer(
            hps,
            text=text,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
            sid=sid,
            language=language,
            device=device,
        )
        torch.cuda.empty_cache()
    return audio


def main():
    args = get_args()
    print(args)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    hps = utils.get_hparams_from_file(args.config)

    if (
        "use_mel_posterior_encoder" in hps.model.keys()
        and hps.model.use_mel_posterior_encoder is True
    ):
        print("Using mel posterior encoder for VITS2")
        posterior_channels = 80  # vits2
        hps.data.use_mel_posterior_encoder = True
    else:
        print("Using lin posterior encoder for VITS1")
        posterior_channels = hps.data.filter_length // 2 + 1
        hps.data.use_mel_posterior_encoder = False

    global net_g
    net_g = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,  # 0 is kept for unknown speaker
        **hps.model)
    net_g = net_g.to(device)

    net_g.eval()
    utils.load_checkpoint(args.checkpoint, net_g, None, skip_optimizer=True)

    sdp_ratio = 0.2
    noise_scale = 0.667
    noise_scale_w = 0.8
    length_scale = 1.0
    speaker_ids = hps.data.spk2id
    languages = ["ZH", "JP", "EN"]
    with open(args.test_file) as fin:
        for line in fin:
            arr = line.strip().split("|")
            audio_path = arr[0]
            if len(arr) >= 4:
                sid = speaker_ids[arr[1]]
                lang = arr[2]
                text = arr[3]

            print(audio_path)
            st = time.time()
            audio = tts_fn(hps, text, sid=sid, sdp_ratio=sdp_ratio,
                           noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                           length_scale=length_scale, language=lang, device=device)

            audio *= 32767 / max(0.01, np.max(np.abs(audio))) * 0.6
            print('RTF {}'.format(
                (time.time() - st) /
                (audio.shape[0] / hps.data.sampling_rate)))
            sys.stdout.flush()
            audio = np.clip(audio, -32767.0, 32767.0)
            wavfile.write(args.outdir + "/" + audio_path.split("/")[-1],
                          hps.data.sampling_rate, audio.astype(np.int16))


if __name__ == '__main__':
    main()

