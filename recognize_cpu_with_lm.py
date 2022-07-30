import torch
import torchaudio
import multiprocessing

from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor


if __name__ == '__main__':

    labels = ["'", "-", "", "\u2047", " ", "\u0430", "\u0431", "\u0432", "\u0433", "\u0434", "\u0435", "\u0436", "\u0437", "\u0438", "\u0439", "\u043a", "\u043b", "\u043c", "\u043d", "\u043e", "\u043f", "\u0440", "\u0441", "\u0442", "\u0443", "\u0444", "\u0445", "\u0446", "\u0447", "\u0448", "\u0449", "\u044c", "\u044e", "\u044f", "\u0454", "\u0456", "\u0457", "\u0491", "<s>", "</s>"]

    device = 'cpu'
    traced_file = './traced_model_cpu.pth'
    vocab_file = './uk/vocab.json'
    test_file = './uk/wavs/uk_1.wav'
    unigrams_file = './uk/small-lm/unigrams.txt'
    kenlm_model_path = './uk/small-lm/5gram.arpa'

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

    # recognize

    with torch.no_grad():

        waveform, _ = torchaudio.load(test_file)
        speech = waveform.squeeze().numpy()

        input_values = processor(speech, return_tensors='pt', padding='longest', sampling_rate=16000).input_values

        logits = model(input_values)[0].numpy()

        decoder = build_ctcdecoder(
            labels,
            kenlm_model_path=kenlm_model_path,
            unigrams=unigrams,
            alpha=0.5,
            beta=1.5,
            unk_score_offset=-10.0,
            lm_score_boundary=True,
        )

        with multiprocessing.get_context("fork").Pool() as pool:
            prediction = decoder.decode_batch(pool, logits)
            
            print(prediction[0])
