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
    elif format == 'onnx':
        dummy_input = torch.randn(1, *model.input_shape)
        torch.onnx.export(
            model, dummy_input, path + suffix,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        )
