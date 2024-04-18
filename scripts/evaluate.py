import torch
import argparse
import sys
import os
import numpy as np
from tqdm import tqdm
import cv2


def tensor_to_image(tensor: torch.Tensor) -> np.array:
    return cv2.normalize(tensor.permute(1, 2, 0).cpu().detach().numpy(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


sys.path.append(os.path.abspath("../src"))
sys.path.append(os.path.abspath(".."))

from src.video_lightning_module import VideoSRLightningModule
from src.dataset import VideoMultiFrameOFDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", type=str, required=True, help="Path to the root directory of the data")
    parser.add_argument("--weights-path", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = VideoSRLightningModule.load_from_checkpoint(args.weights_path, map_location=device)

    dataset = VideoMultiFrameOFDataset(args.root_path + "/frames", None, args.root_path + "/flows", num_frames=module.num_frames)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    module.eval()

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = {k: v.to(module.device) for k, v in batch.items()}
        bilinear = torch.nn.functional.upsample_bilinear(batch["LQ"].squeeze(0)[1].unsqueeze(0), scale_factor=4).squeeze(0)
        hq = module.predict_step(batch, idx).squeeze(0)
        print(bilinear.shape)
        print(hq.shape)

        bilinear = tensor_to_image(bilinear)
        hq = tensor_to_image(hq)

        stacked_frames = np.hstack([bilinear, hq])

        cv2.imshow("Combined Video", stacked_frames)

        # Wait for 1 ms before moving to the next frame, and break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
