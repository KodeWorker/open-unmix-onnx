# Open-Unmix Onnx/OpenVino Runtime Support
This repository is a workaround for exporting Open-Unmix as ONNX models.
We export ```OpenUnmix``` models and implement ```Separator``` using non-pytorch approach. 

# Instructions
Please read environment.yml for information on dependencies.
You may change pytorch/cuda version to meet your system.

1. Build conda environment
```
conda env create -f environment.yml
```
2. activate umx environment
```
conda activate umx
```
3. Export UMXL models
```
python export_umx_to_onnx.py --output_dur ./model
```
There will export four onnx models in ./model.
4. Prepare your mp3 file (e.g. ./source/Music.mp3)
5. Run UMXL music source separation
```
python run_umx_openvino.py --input_file ./source/Music.mp3 --output_dir ./output
```
There will generate four tracks of mp3 file in ./output/Music.

# Know issues
1. UMXL model can be exported with dynamic shape, however we cannot send dynamic-shaped inputs which would cause error.
   Size of UMXL model is fixed on ```(1,2,44100)```, which is two-channel signals with 1 second duration.

# Reference
- Open-Unmix
https://github.com/sigsep/open-unmix-pytorch