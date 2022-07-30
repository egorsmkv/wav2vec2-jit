import torch
import torchaudio

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor


if __name__ == '__main__':

    device = 'cpu'
    traced_file = './traced_model_cpu.pth'
    hf_model_name = 'Yehor/wav2vec2-xls-r-300m-uk-with-small-lm'

    test_file = './wavs/uk_1.wav'

    model = torch.jit.load(traced_file)
    model = model.to(device)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hf_model_name)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(hf_model_name)
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

