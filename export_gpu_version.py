import pathlib

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor


def load_model(hf_model, device):
    model = Wav2Vec2ForCTC.from_pretrained(hf_model).to(device)
    model.config.return_dict = False
    model.eval()

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hf_model)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(hf_model)
    processor = Wav2Vec2Processor(feature_extractor, tokenizer)

    return model, processor


def trace_torchscript_model(hf_model, device='cpu'):
    current_dir = pathlib.Path(__file__).parent
    output_file = f'{current_dir}/traced_model_{device}.pth'

    model, _ = load_model(hf_model, device)

    with torch.autocast(device) and torch.no_grad():
        traced_model = torch.jit.trace(model, (torch.randn((1,16000), device=device)))

    torch.jit.save(traced_model, output_file)

    print("Exported.")

    return model


if __name__ == '__main__':

    device_type = 'cuda'
    hf_model_name = 'Yehor/wav2vec2-xls-r-300m-uk-with-small-lm'

    trace_torchscript_model(hf_model_name, device_type)
