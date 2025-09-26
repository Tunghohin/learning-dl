import torch

MODEL_FORMAT_SUFFIX = {
    "onnx": ".onnx",
    "torch": ".pt",
}

def export_model(model, path, format):
    if format not in MODEL_FORMAT_SUFFIX:
        raise ValueError(f"Unsupported format: {format}. Supported formats: {list(MODEL_FORMAT_SUFFIX.keys())}")

    suffix = MODEL_FORMAT_SUFFIX[format]
    if format == 'torch':
        torch.save(model.state_dict(), path + suffix)
    else:
        raise NotImplementedError(f"Export format {format} is not implemented yet.")