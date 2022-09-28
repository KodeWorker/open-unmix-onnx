import os
import numpy as np
from umx_openvino import UMX_openvino, Separator_openvino
import pydub
import argparse
import numpy as np
from tqdm import tqdm

def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y

def write(f, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="mp3 audio file", required=True)
    parser.add_argument("--output_dir", help="directory to store tracks", required=True)
    args = parser.parse_args()
    
    input_file=args.input_file
    output_dir=os.path.join(args.output_dir, os.path.basename(input_file).split(".")[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    rate, audio = read(input_file, normalized=True)
    audio = audio.transpose(1, 0)
    audio = np.expand_dims(audio, axis=0)
    # resample
    n_segsize = 44100
    
    bass_umx = UMX_openvino("CPU", os.path.join("model", "bass.onnx"))
    drums_umx = UMX_openvino("CPU", os.path.join("model", "drums.onnx"))
    other_umx = UMX_openvino("CPU", os.path.join("model", "other.onnx"))
    vocals_umx = UMX_openvino("CPU", os.path.join("model", "vocals.onnx"))
    
    separator = Separator_openvino({"bass": bass_umx,
                                    "drums": drums_umx,
                                    "other": other_umx,
                                    "vocals": vocals_umx}, niter=1)
    
    estimates_list = []
    for i in tqdm(range(0, audio.shape[2], n_segsize)):
        inputs = np.zeros([1,2,n_segsize])
        copy_tensor = audio[:,:,i:i+n_segsize]
        inputs[:,:,:copy_tensor.shape[2]] = copy_tensor
        estimates = separator.inference(inputs)
        estimates_list.append(estimates)
    estimates_ = np.concatenate(estimates_list, axis=3)
    estimates = separator.to_dict(estimates_, aggregate_dict=None)

    for target, estimate in estimates.items():
        outputs = np.squeeze(estimate, axis=0)
        outputs = outputs.transpose(1, 0)
        target_path = os.path.join(output_dir, "{}.mp3".format(target))
        write(target_path, rate, outputs, normalized=True)