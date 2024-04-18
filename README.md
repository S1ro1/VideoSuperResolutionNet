# Video Super Resolution using Optical Flow and Deformable Convolutions

This repository contains the code for my Bachelor's Thesis on Video Super Resolution using Optical Flow and Deformable Convolutions. The thesis is available TODO

To install the required dependencies, run `pip install -r requirements.txt`. The code was tested with Python 3.10.12. After installing dependencies, installing mmcv is required with `mim install mmcv`

Source code for training is available in the `src` directory, and the code for evaluation is available in the `eval` directory. To download the weights for the models run `bash download_weights.sh` and to download the weights for the RAFT model run `bash scripts/RAFT/download_models.sh`.

## Evaluation

To prepare a video for evaluation do the following inside the scripts directory:

```bash
$ python3 prepare_video.py --video-path /path/to/video.mp4 --output-path /path/to/output-folder --raft-weights /path/to/raft-weights.pth
```

To evaluate the video run the following inside the scripts directory:

```bash
$ python3 evaluate.py --root-path /path/to/output-folder --weights-path /path/to/weights.ckpt --output-path /path/to/output-folder --visualize=True
```

Visualizations are disabled by default, such as saving the results to output folder. To enable visualizations, set `--visualize=True`.


