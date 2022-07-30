import pathlib

import torch
from transformers import Wav2Vec2ForCTC


def load_model(hf_model, device):
    model = Wav2Vec2ForCTC.from_pretrained(hf_model).to(device)
    model.config.return_dict = False
    model.eval()

    return model


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
