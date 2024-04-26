import sys
import os

sys.path.append(os.path.join("RAFT", "core"))

from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import InputPadder
import torchvision
import torch
import pathlib
import argparse
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=str, required=True, help="Path to the video file")
    parser.add_argument("--output-path", type=str, required=True, default="example_output", help="Path to the output directory")
    parser.add_argument("--raft-weights", type=str, default="raft-kitti.pth", help="Path to the RAFT model weights")
    parser.add_argument("--devices", type=int, nargs="+", default=[0], help="Devices to use")
    return parser.parse_args()


@torch.no_grad()
def prepare_video(video_path: str, output_path: str, model: torch.nn.Module):
    name = pathlib.Path(video_path).stem
    frame_output_path = output_path + "/frames/" + name
    flow_output_path = output_path + "/flows/" + name
    os.makedirs(frame_output_path, exist_ok=True)
    os.makedirs(flow_output_path, exist_ok=True)
    model.eval()

    vframes = (
        (torchvision.io.read_video(video_path, output_format="TCHW", pts_unit="sec")[0])
        .float()
        .to("cuda" if torch.cuda.is_available() else "cpu")
    )
    iterator = tqdm(enumerate(zip(vframes[:], vframes[1:], vframes[2:]), start=1), desc="Processing video frames", total=len(vframes) - 2)

    for index, vframe in enumerate(vframes):
        torchvision.utils.save_image(vframe / 255.0, f"{frame_output_path}/{index:06d}.png")

    for index, (first_frame, second_frame, third_frame) in iterator:
        padder = InputPadder(first_frame.shape)
        padded_frames = [x.unsqueeze(0) for x in padder.pad(first_frame, second_frame, third_frame)]
        _, flow_up = model(padded_frames[1], padded_frames[2], iters=20, test_mode=True)
        _, flow_down = model(padded_frames[1], padded_frames[0], iters=20, test_mode=True)
        torch.save(flow_up, f"{flow_output_path}/{index:06d}_up.pt")
        torch.save(flow_down, f"{flow_output_path}/{index:06d}_down.pt")


def main():
    args = parse_args()
    args.small = False
    args.mixed_precision = False

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(RAFT(args), device_ids=args.devices)
        model.load_state_dict(torch.load(args.raft_weights))
        model = model.module
    else:
        model = RAFT(args)
        weights = torch.load(args.raft_weights, map_location="cpu")
        weights = {k.replace("module.", ""): v for k, v in weights.items()}
        model.load_state_dict(weights)

    prepare_video(args.video_path, args.output_path, model)


if __name__ == "__main__":
    main()
