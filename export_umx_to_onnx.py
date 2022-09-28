import os
import argparse
import torch
import torch.onnx
from openunmix import umxl_spec
from openunmix.utils import preprocess
from openunmix.transforms import TorchSTFT, ComplexNorm
from openunmix.transforms import make_filterbanks

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", help="directory to save onnx models", required=True)
    args = parser.parse_args()

    nb_samples = 1
    nb_channels = 2
    nb_timesteps = 44100 # 44100 data for 1 sec 
    model_dir = args.output_dir
    
    audio = torch.rand((nb_samples, nb_channels, nb_timesteps), device="cpu")
    target_models = umxl_spec(targets=None, device="cpu", pretrained="True")
    
    audio = preprocess(audio)
    
    stft, _ = make_filterbanks(
        n_fft=4096,
        n_hop=1024,
        center=True,
        method="torch",
        sample_rate=44100.0,
    )
    complexnorm = ComplexNorm(mono=nb_channels == 1)
    mix_stft = stft(audio)
    X = complexnorm(mix_stft)
    print(X.shape)
    
    dynamic_axes = {
    "input": {3: "nb_segments"},
    "output": {3: "nb_segments"}
    }
    
    for model_name in target_models.keys():
        
        onnx_path = os.path.join(model_dir, "{}.onnx".format(model_name))
        
        # Export the model
        torch.onnx.export(
            target_models[model_name],
            X,
            onnx_path,
            export_params=True,
            opset_version=15,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes
        )