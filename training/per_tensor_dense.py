"""Rewrite an INT8 .tflite so each FULLY_CONNECTED layer uses one weight scale instead of one per output.

OpenMV firmware 5.0.1 only supports per-tensor fully connected layers and rejects per-channel ones with
"Failed to allocate tensors". The file is patched in place: the scale vectors shrink to one entry, and the
weights and bias are requantized to the largest per-output scale, so every other byte stays the same.
"""

import argparse
import struct

import numpy as np
import tflite
from tflite.BuiltinOperator import BuiltinOperator

# Field offsets from the TFLite flatbuffer schema.
SCALE_FIELD = 8
ZERO_POINT_FIELD = 10
DATA_FIELD = 4


def vector_start(table, field):
    offset = table._tab.Offset(field)
    if offset == 0:
        raise ValueError("Missing flatbuffer field")
    return table._tab.Vector(offset)


def set_single_scale(data, quantization, scale):
    scales = vector_start(quantization, SCALE_FIELD)
    zero_points = vector_start(quantization, ZERO_POINT_FIELD)
    struct.pack_into("<I", data, scales - 4, 1)
    struct.pack_into("<f", data, scales, scale)
    struct.pack_into("<I", data, zero_points - 4, 1)
    struct.pack_into("<q", data, zero_points, 0)


def convert(data):
    model = tflite.Model.GetRootAsModel(data, 0)
    graph = model.Subgraphs(0)
    converted = 0
    for op_index in range(graph.OperatorsLength()):
        op = graph.Operators(op_index)
        code = model.OperatorCodes(op.OpcodeIndex())
        if max(code.BuiltinCode(), code.DeprecatedBuiltinCode()) != BuiltinOperator.FULLY_CONNECTED:
            continue
        input_index, weights_index, bias_index = op.InputsAsNumpy()[:3]
        weights = graph.Tensors(weights_index)
        weight_scales = weights.Quantization().ScaleAsNumpy().astype(np.float64)
        if len(weight_scales) == 1:
            continue
        if bias_index < 0:
            raise ValueError("Fully connected layer %d has no bias" % op_index)
        bias = graph.Tensors(bias_index)
        input_scale = np.float32(graph.Tensors(input_index).Quantization().ScaleAsNumpy()[0])

        new_scale = np.float32(weight_scales.max())
        ratio = weight_scales / np.float64(new_scale)
        weight_buffer = model.Buffers(weights.Buffer())
        weight_start = vector_start(weight_buffer, DATA_FIELD)
        rows = len(weight_scales)
        old_weights = np.frombuffer(data, np.int8, weight_buffer.DataLength(), weight_start).reshape(rows, -1)
        new_weights = np.clip(np.round(old_weights * ratio[:, None]), -127, 127).astype(np.int8)
        data[weight_start:weight_start + new_weights.nbytes] = new_weights.tobytes()

        bias_buffer = model.Buffers(bias.Buffer())
        bias_start = vector_start(bias_buffer, DATA_FIELD)
        old_bias = np.frombuffer(data, "<i4", rows, bias_start)
        new_bias = np.round(old_bias * ratio).astype("<i4")
        data[bias_start:bias_start + new_bias.nbytes] = new_bias.tobytes()

        set_single_scale(data, weights.Quantization(), new_scale)
        set_single_scale(data, bias.Quantization(), np.float32(input_scale * new_scale))
        converted += 1
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    with open(args.source, "rb") as source:
        data = bytearray(source.read())
    converted = convert(data)
    with open(args.destination, "wb") as destination:
        destination.write(data)
    print("Converted %d fully connected layer(s) to per-tensor quantization" % converted)


if __name__ == "__main__":
    main()
