import onnxruntime as ort
import numpy as np

providers = [
    (
        'CUDAExecutionProvider',
        {
           'device_id': 0,
        }
    ),
    'CPUExecutionProvider',
]

session = ort.InferenceSession('test_model.onnx', providers=providers)

X = np.random.rand(10).astype(np.float16)

output_names = ["Y"]

use_iobinding = True

if use_iobinding:
    binding = session.io_binding()
    binding.bind_cpu_input('X', X)
    binding.bind_output('Y', device_type='cuda', device_id=0)

    binding.synchronize_inputs()
    session.run_with_iobinding(binding)
    binding.synchronize_outputs()

    results = {
        name: output.numpy()
        for name, output
        in zip(output_names, binding.get_outputs())
    }
else:
    inputs = {
        "X": X,
    }
    results = {
        name: output
        for name, output
        in zip(output_names, session.run(output_names, inputs))
    }
