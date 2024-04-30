# Video Super Resolution using Optical Flow and Deformable Convolutions

This repository contains the code for my Bachelor's Thesis on Video Super Resolution using Optical Flow and Deformable Convolutions. The thesis is available TODO

To install the required dependencies, run `pip install -r requirements.txt`. The code was tested with Python 3.10.12 on Linux and Python 3.9.15 on Windows 11. After installing dependencies, installing mmcv is required with `mim install mmcv-full`

Source code for training is available in the `src` directory, and the code for evaluation is available in the `scripts` directory. To download the weights for the models run `bash download_weights.sh` which will create a Weights folder in the root folder. To download the weights for the RAFT model run `bash download_models.sh` inside `scripts/RAFT` directory, which will create a models folder inside.

## Evaluation

To prepare a video for evaluation do the following inside the `scripts` directory:

```bash
$ python3 prepare_video.py --video-path /path/to/video.mp4 --output-path /path/to/output-folder --raft-weights /path/to/raft-weights.pth
```

To evaluate the video run the following inside the `scripts` directory:

```bash
$ python3 evaluate.py --root-path /path/to/output-folder --weights-path /path/to/weights.ckpt --output-path /path/to/output-folder
```

Visualizations are disabled by default, such as saving the results to output folder. To enable visualizations, set `--visualize`.


## Training

To train the model, run the following inside the `src` directory:

```bash
$ python3 main.py --config /path/to/config-file.json
```

Example configuration files are available in the `configs` directory. Only logging with Weights and Biases is properly tested, and the code will not work without a valid Weights and Biases account. Untested logging with Tensorboard is available by setting `logger: Tensorboard` in the configuration file. Configuration files are plug-and-play and the only required change is to change devices and paths to the dataset.

Model was trained on [REDS](https://seungjunnah.github.io/Datasets/reds.html) dataset. The optical flow dataset can be then prepared by running the following inside the `src/scripts/RAFT` directory:

```bash
$ python3 demo.py --model /path/to/raft-weights.pth --outdir /path/to/output-folder/{val,train} --dataset-path /path/to/REDS/{val,train}
```

Script has to be run for both train and val datasets. The output folder will contain the optical flows for each sequence, and then this folder can be used as the dataset path in the configuration file as such:
    
```json
{
    "train_dataset_args": {
        "hq_path": "/path/to/REDS/train/train_sharp",
        "lq_path": "/path/to/REDS/train/train_sharp_bicubic/X4",
        "of_path": "/path/to/output-folder/train",
        "num_frames": 3
    },
    "val_dataset_args": {
        "hq_path": "/path/to/REDS/val/val_sharp",
        "lq_path": "/path/to/REDS/val/val_sharp_bicubic/X4",
        "of_path": "/path/to/output-folder/val",
        "num_frames": 3
    }
}
```

For some datasets, the optical flow or num_frames is optional, all possible argument combinations are to be seen in the `src/datasets.py` file. Correct dataset will be chosen based on the model configuration.

To train code on different datasets, other than REDS, the dataset has to be prepared in the same structure as REDS, and the configuration file has to be changed accordingly, such as padding etc.
