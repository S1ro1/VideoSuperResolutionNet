# Author: Matej Sirovatka

import torch
import torchvision
import argparse
import sys
import os
import numpy as np
from tqdm import tqdm


sys.path.append(os.path.abspath("../src"))
sys.path.append(os.path.abspath(".."))

from src.video_lightning_module import VideoSRLightningModule
from src.dataset import VideoMultiFrameOFDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", type=str, required=True, help="Path to the root directory of the data")
    parser.add_argument("--output-path", default=None, help="Store the output, if provided")
    parser.add_argument("--weights-path", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--visualize", action="store_true", help="Visualize the output")
    return parser.parse_args()


def main():
    args = parse_args()
    print(args.root_path)
    if args.visualize:
        import cv2

        def tensor_to_image(tensor: torch.Tensor) -> np.array:
            return cv2.normalize(tensor.permute(1, 2, 0).cpu().detach().numpy(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = VideoSRLightningModule.load_from_checkpoint(args.weights_path, map_location=device)
    print(f"Loaded model from {args.weights_path} to device {device}")

    num_frames = module.hparams.get("num_frames")
    if num_frames == 1:
        from src.dataset import VideoSingleFrameDataset

        dataset = VideoSingleFrameDataset(args.root_path + "/frames", args.root_path + "/frames")
    else:
        dataset = VideoMultiFrameOFDataset(args.root_path + "/frames", None, args.root_path + "/flows", num_frames=module.num_frames)
    print(f"Loaded dataset with {len(dataset)} samples")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    module.eval()
    if args.output_path:
        print(f"Saving output to {args.output_path}")
        print("Warning: Saving the output makes the evaluation significantly slower")
        os.makedirs(args.output_path, exist_ok=True)

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = {k: v.to(module.device) for k, v in batch.items()}
        if num_frames == 1:
            bilinear = torch.nn.functional.interpolate(batch["LQ"], scale_factor=4, mode="bilinear").squeeze(0)
        else:
            bilinear = torch.nn.functional.interpolate(batch["LQ"].squeeze(0)[1].unsqueeze(0), scale_factor=4, mode="bilinear").squeeze(0)
        hq = module.predict_step(batch, idx).squeeze(0)

        if args.output_path:
            os.makedirs(args.output_path, exist_ok=True)
            torchvision.utils.save_image(hq, f"{args.output_path}/{idx:06d}.png")
            torchvision.utils.save_image(bilinear, f"{args.output_path}/{idx:06d}_bilinear.png")

        if not args.visualize:
            continue

        bilinear = tensor_to_image(bilinear)
        hq = tensor_to_image(hq)

        stacked_frames = np.hstack([bilinear, hq])

        caption_height = 30
        caption_image = np.zeros((caption_height, stacked_frames.shape[1], 3), dtype=np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_color = (255, 255, 255)

        cv2.putText(caption_image, "Bilinear 4x", (10, 20), font, font_scale, text_color, font_thickness)
        cv2.putText(caption_image, "Learned 4x", (stacked_frames.shape[1] // 2 + 10, 20), font, font_scale, text_color, font_thickness)

        image = np.vstack([stacked_frames, caption_image])
        cv2.imshow("Biliear vs Learned", image)

        # Wait for 1 ms before moving to the next frame, and break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if args.visualize:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
