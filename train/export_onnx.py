import argparse
import torch

from model import PoseNet
from dataset import NUM_JOINTS, IMG_SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="train/pose_net.pt")
    ap.add_argument("--out", default="frontend/pose-tracker/vendor/pose_net.onnx")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = PoseNet(NUM_JOINTS)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["heatmaps"],
        dynamic_axes={"input": {0: "batch"}, "heatmaps": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported to {args.out} (epoch={ckpt['epoch']}, val_pck={ckpt['val_pck']:.3f})")

    import onnx
    import onnxruntime as ort
    import numpy as np

    onnx_model = onnx.load(args.out)
    onnx.checker.check_model(onnx_model)

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": dummy.numpy()})
    print("onnxruntime output shape:", out[0].shape)

    with torch.no_grad():
        torch_out = model(dummy).numpy()
    diff = np.abs(torch_out - out[0]).max()
    print(f"max abs diff vs torch: {diff:.6f}")


if __name__ == "__main__":
    main()
