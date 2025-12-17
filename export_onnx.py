# export_onnx.py
import torch
import json
from torchvision import models
import argparse
from pathlib import Path

def build_model(num_classes, device):
    model = models.efficientnet_b0(pretrained=False)
    in_f = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_f, num_classes)
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_pt", type=str, default="model.pt")
    p.add_argument("--onnx_out", type=str, default="model.onnx")
    p.add_argument("--dataset_dir", type=str, default="dataset")
    p.add_argument("--img_size", type=int, default=224)
    args = p.parse_args()

    # Get class list from dataset/train folder
    import os
    train_dir = os.path.join(args.dataset_dir, "train")
    classes = sorted(os.listdir(train_dir))
    num_classes = len(classes)
    print("Classes:", classes)

    device = torch.device("cpu")
    model = build_model(num_classes, device)
    state = torch.load(args.model_pt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # dummy input
    dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)

    torch.onnx.export(
        model,
        dummy,
        args.onnx_out,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print("Exported ONNX to", args.onnx_out)

    # save classes
    with open("classes.json", "w") as f:
        json.dump(classes, f)
    print("Saved classes.json")

if __name__ == "__main__":
    main()
