import torch
import torchaudio
import multiprocessing
from glob import glob
from os.path import exists

from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor


def load_lm(labels, kenlm_model_path, unigrams):
    return build_ctcdecoder(
            labels,
            kenlm_model_path=kenlm_model_path,
            unigrams=unigrams,
            alpha=0.5,
            beta=1.5,
            unk_score_offset=-10.0,
            lm_score_boundary=True,
    )


def load_model():
    labels = ["'", "-", "", "\u2047", " ", "\u0430", "\u0431", "\u0432", "\u0433", "\u0434", "\u0435", "\u0436", "\u0437", "\u0438", "\u0439", "\u043a", "\u043b", "\u043c", "\u043d", "\u043e", "\u043f", "\u0440", "\u0441", "\u0442", "\u0443", "\u0444", "\u0445", "\u0446", "\u0447", "\u0448", "\u0449", "\u044c", "\u044e", "\u044f", "\u0454", "\u0456", "\u0457", "\u0491", "<s>", "</s>"]

    device = 'cuda'
    traced_file = './traced_model_cuda.pth'
    vocab_file = './uk/vocab.json'
    unigrams_file = './uk/news-lm/unigrams.txt'
    kenlm_model_path = './uk/news-lm/news_3gram_correct.bin'

    unigrams = []
    with open(unigrams_file, 'r') as f:
        unigrams = [it.strip() for it in f.readlines()]

    model = torch.jit.load(traced_file)
    model = model.to(device)

    feature_extractor = Wav2Vec2FeatureExtractor(
        do_normalize=True,
        feature_size=1,
        padding_side='right',
        padding_value=0.0,
        return_attention_mask=True,
        sampling_rate=16000
    )
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        replace_word_delimiter_char=" ",
        do_lower_case=False,    
    )
    processor = Wav2Vec2Processor(feature_extractor, tokenizer)

    lm_decoder = load_lm(labels, kenlm_model_path, unigrams)

    return model, processor, lm_decoder


def recognize(model, processor, lm_decoder, pool, filename):
    with torch.no_grad():
        waveform, sr = torchaudio.load(filename)
        if sr != 16_000:
            raise ValueError(f'WARN: wrong SR: {sr}')

        speech = waveform.squeeze().numpy()

        input_values = processor(speech, return_tensors='pt', padding='longest', sampling_rate=16000).input_values
        input_values = input_values.cuda()

        logits = model(input_values)[0].cpu().numpy()

        prediction = lm_decoder.decode_batch(pool, logits)

        return prediction[0]

################
# RUN
################

_model, _processor, _lm_decoder = load_model()

with multiprocessing.get_context("fork").Pool(processes=8) as pool:
    for filename in glob('/home/yehor/ext-disk/VOA-dataset/chunks/**/*.wav'):
        transcription_file = filename + '.txt'

        if exists(transcription_file):
            print(f'File {filename} is recognized')
            continue

        try:
            transcription = recognize(_model, _processor, _lm_decoder, pool, filename)

            with open('transcriptions.txt', 'a') as f:
                row = f'{filename},{transcription}\n'
                f.write(row)
            
            with open(transcription_file, 'w') as f:
                row = f'{transcription}'
                f.write(row)

            print(f'{filename},{transcription}')

        except ValueError:
            print(f'File {filename} is not recognized, error happened')
