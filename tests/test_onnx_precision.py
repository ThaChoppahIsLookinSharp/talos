from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import onnx
from onnx import TensorProto, helper

from talos.evaluation.zigzag_evaluator import (
    ZIGZAG_ONNX_OPERATORS,
    ZIGZAG_PRECISION_ATTRIBUTES,
    prepare_onnx_workload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _precision_attributes(node: onnx.NodeProto) -> dict[str, int]:
    return {
        attribute.name: int(helper.get_attribute_value(attribute))
        for attribute in node.attribute
        if attribute.name in ZIGZAG_PRECISION_ATTRIBUTES
    }


class OnnxPrecisionTests(unittest.TestCase):
    def test_float32_workloads_use_visible_onnx_types(self) -> None:
        for relative_path in (
            "workloads/alexnet.onnx",
            "workloads/resnet18_first_layer.onnx",
        ):
            with self.subTest(workload=relative_path):
                path = REPO_ROOT / relative_path
                original_bytes = path.read_bytes()

                prepared, formats = prepare_onnx_workload(path)

                compute_nodes = [
                    node
                    for node in prepared.graph.node
                    if node.op_type in ZIGZAG_ONNX_OPERATORS
                ]
                self.assertTrue(compute_nodes)
                self.assertEqual(len(formats), len(compute_nodes))
                for node in compute_nodes:
                    self.assertEqual(
                        _precision_attributes(node),
                        {
                            "act_size": 32,
                            "weight_size": 32,
                            "output_size": 32,
                        },
                    )
                self.assertTrue(
                    all(
                        layer_formats
                        == {
                            "I": "float32",
                            "W": "float32",
                            "O": "float32",
                        }
                        for layer_formats in formats.values()
                    )
                )
                self.assertEqual(path.read_bytes(), original_bytes)

                original = onnx.load(path, load_external_data=False)
                self.assertTrue(
                    all(
                        not _precision_attributes(node)
                        for node in original.graph.node
                    )
                )

    def test_external_initializer_data_is_not_required(self) -> None:
        path = REPO_ROOT / "workloads" / "alexnet.onnx"
        self.assertFalse(
            (path.parent / "external_data_filename_test").exists()
        )

        _prepared, formats = prepare_onnx_workload(path)

        self.assertTrue(formats)

    def test_int8_uses_eight_bits_for_visible_output(self) -> None:
        input_info = helper.make_tensor_value_info(
            "input",
            TensorProto.INT8,
            [1, 2],
        )
        output_info = helper.make_tensor_value_info(
            "output",
            TensorProto.INT8,
            [1, 1],
        )
        weight = helper.make_tensor(
            "weight",
            TensorProto.INT8,
            [2, 1],
            [1, 1],
        )
        node = helper.make_node(
            "MatMul",
            ["input", "weight"],
            ["output"],
        )
        model = helper.make_model(
            helper.make_graph(
                [node],
                "int8_matmul",
                [input_info],
                [output_info],
                [weight],
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "int8.onnx"
            onnx.save(model, path)
            prepared, formats = prepare_onnx_workload(path)

        self.assertEqual(
            _precision_attributes(prepared.graph.node[0]),
            {
                "act_size": 8,
                "weight_size": 8,
                "output_size": 8,
            },
        )
        self.assertEqual(
            formats,
            {0: {"I": "int8", "W": "int8", "O": "int8"}},
        )

    def test_unsupported_tensor_type_fails_clearly(self) -> None:
        tensor_info = helper.make_tensor_value_info(
            "input",
            TensorProto.STRING,
            [1, 1],
        )
        output_info = helper.make_tensor_value_info(
            "output",
            TensorProto.STRING,
            [1, 1],
        )
        weight = helper.make_tensor(
            "weight",
            TensorProto.STRING,
            [1, 1],
            ["value"],
        )
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "MatMul",
                        ["input", "weight"],
                        ["output"],
                    )
                ],
                "unsupported",
                [tensor_info],
                [output_info],
                [weight],
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unsupported.onnx"
            onnx.save(model, path)
            with self.assertRaisesRegex(
                ValueError,
                "Unsupported ONNX tensor type 'STRING'",
            ):
                prepare_onnx_workload(path)


if __name__ == "__main__":
    unittest.main()
