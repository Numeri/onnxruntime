import onnx
from onnx import helper
import numpy as np

input_node = helper.make_tensor_value_info(
    name='X',
    elem_type=helper.np_dtype_to_tensor_dtype(np.dtype('float16')),
    shape=('seq_len',),
)
output_node = helper.make_tensor_value_info(
    name='Y',
    elem_type=helper.np_dtype_to_tensor_dtype(np.dtype('float16')),
    shape=('seq_len',),
)
softmax_node = helper.make_node(
    "MyTritonKernel",
    inputs=["X"],
    outputs=["Y"],
    domain="com.microsoft",
)

graph = helper.make_graph(
    name='test_model',
    nodes=[softmax_node],
    inputs=[input_node],
    outputs=[output_node],
)

model = helper.make_model(graph=graph)
onnx.save(model, 'test_model.onnx')
