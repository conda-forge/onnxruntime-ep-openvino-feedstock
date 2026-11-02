import os
import sys

import numpy as np
import onnxruntime as ort

FLOAT = 1
INTS = 7


def varint(value):
    out = bytearray()
    while value > 0x7F:
        out.append(value & 0x7F | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def int_field(number, value):
    return varint(number << 3) + varint(value)


def bytes_field(number, value):
    if isinstance(value, str):
        value = value.encode()
    return varint(number << 3 | 2) + varint(len(value)) + value


def make_tensor(name, array):
    return (
        b"".join(int_field(1, dim) for dim in array.shape)
        + int_field(2, FLOAT)
        + bytes_field(8, name)
        + bytes_field(9, array.tobytes())
    )


def make_ints_attribute(name, values):
    return (
        bytes_field(1, name)
        + b"".join(int_field(8, value) for value in values)
        + int_field(20, INTS)
    )


def make_node(op_type, inputs, outputs, attributes=()):
    return (
        b"".join(bytes_field(1, name) for name in inputs)
        + b"".join(bytes_field(2, name) for name in outputs)
        + bytes_field(4, op_type)
        + b"".join(bytes_field(5, attribute) for attribute in attributes)
    )


def make_value_info(name, shape):
    dims = b"".join(bytes_field(1, int_field(1, dim)) for dim in shape)
    tensor_type = int_field(1, FLOAT) + bytes_field(2, dims)
    return bytes_field(1, name) + bytes_field(2, bytes_field(1, tensor_type))


device = sys.argv[1] if len(sys.argv) > 1 else "CPU"
if sys.platform == "win32":
    lib = os.path.join(sys.prefix, "Library", "bin", "onnxruntime_providers_openvino_plugin.dll")
else:
    lib = os.path.join(sys.prefix, "lib", "libonnxruntime_providers_openvino_plugin.so")
ort.register_execution_provider_library("OpenVINO", lib)

ov_devices = [d for d in ort.get_ep_devices() if d.ep_name == "OpenVINOExecutionProvider"]
print("OpenVINO EP devices:", [(d.device.type.name, dict(d.ep_metadata)) for d in ov_devices])
if not ov_devices:
    vendors = {hex(d.device.vendor_id) for d in ort.get_ep_devices()}
    assert "0x8086" not in vendors, vendors
    print(f"No Intel device ({vendors}), skipping inference")
    sys.exit(0)

rng = np.random.default_rng(0)
w = (rng.standard_normal((8, 3, 3, 3)) * 0.1).astype(np.float32)
fc = (rng.standard_normal((8, 4)) * 0.1).astype(np.float32)
nodes = [
    make_node("Conv", ["x", "w"], ["c"], [make_ints_attribute("pads", [1, 1, 1, 1])]),
    make_node("Relu", ["c"], ["r"]),
    make_node("GlobalAveragePool", ["r"], ["p"]),
    make_node("Flatten", ["p"], ["f"]),
    make_node("MatMul", ["f", "fc"], ["y"]),
]
graph = (
    b"".join(bytes_field(1, node) for node in nodes)
    + bytes_field(2, "ov_ep_test")
    + bytes_field(5, make_tensor("w", w))
    + bytes_field(5, make_tensor("fc", fc))
    + bytes_field(11, make_value_info("x", [1, 3, 16, 16]))
    + bytes_field(12, make_value_info("y", [1, 4]))
)
data = int_field(1, 9) + bytes_field(7, graph) + bytes_field(8, bytes_field(1, "") + int_field(2, 17))

x = rng.standard_normal((1, 3, 16, 16)).astype(np.float32)
ref = ort.InferenceSession(data, providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]

chosen = [d for d in ov_devices if d.device.type.name == device]
assert chosen, (device, [d.device.type.name for d in ov_devices])
so = ort.SessionOptions()
so.add_provider_for_devices(chosen[:1], {})
so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
sess = ort.InferenceSession(data, sess_options=so)
out = sess.run(None, {"x": x})[0]
print("providers:", sess.get_providers(), "max abs diff:", float(np.abs(out - ref).max()))
assert "OpenVINOExecutionProvider" in sess.get_providers()
np.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-3)
print(f"OpenVINO EP plugin on {device} OK")
