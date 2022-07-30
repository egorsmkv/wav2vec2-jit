import torch
import torchaudio

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor


if __name__ == '__main__':

    device = 'cpu'
    traced_file = './traced_model_cpu.pth'
    vocab_file = './uk/vocab.json'
    test_file = './uk/wavs/uk_1.wav'

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

        logits = model(input_values)[0]

        pred_ids = torch.argmax(logits, dim=-1)
        prediction = tokenizer.decode(pred_ids[0])

        print(prediction)

