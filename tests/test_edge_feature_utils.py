import importlib.util
import sys
import types
import unittest
from types import SimpleNamespace

import torch

from edge_feature_utils import (
    average_residual_update,
    get_edge_type_index,
    get_non_normalized_edge_feature_indices,
    get_port_feature_indices,
    get_time_delta_feature_indices,
    z_norm_except,
)


if importlib.util.find_spec("torch_geometric") is None:
    fake_data_module = types.ModuleType("torch_geometric.data")
    fake_typing_module = types.ModuleType("torch_geometric.typing")

    class _FakeData:
        def __init__(self, *args, **kwargs):
            pass

    class _FakeStore:
        pass

    class _FakeHeteroData(dict):
        def __getitem__(self, key):
            if key not in self:
                dict.__setitem__(self, key, _FakeStore())
            return dict.__getitem__(self, key)

    fake_data_module.Data = _FakeData
    fake_data_module.HeteroData = _FakeHeteroData
    fake_typing_module.OptTensor = object

    sys.modules.setdefault("torch_geometric", types.ModuleType("torch_geometric"))
    sys.modules["torch_geometric.data"] = fake_data_module
    sys.modules["torch_geometric.typing"] = fake_typing_module

from data_util import create_hetero_obj


class EdgeFeatureUtilsTests(unittest.TestCase):
    def test_average_residual_update_returns_mean(self):
        current = torch.tensor([[2.0, 4.0]])
        update = torch.tensor([[6.0, 8.0]])

        result = average_residual_update(current, update)

        self.assertTrue(torch.equal(result, torch.tensor([[4.0, 6.0]])))

    def test_rgcn_keeps_payment_format_column_unnormalized(self):
        edge_attr = torch.tensor(
            [
                [1.0, 10.0, 100.0, 0.0, 7.0, 70.0, 700.0, 7000.0],
                [2.0, 20.0, 200.0, 1.0, 8.0, 80.0, 800.0, 8000.0],
                [3.0, 30.0, 300.0, 2.0, 9.0, 90.0, 900.0, 9000.0],
            ]
        )

        normalized = z_norm_except(edge_attr, get_non_normalized_edge_feature_indices("rgcn"))

        self.assertTrue(torch.equal(normalized[:, get_edge_type_index()], edge_attr[:, get_edge_type_index()]))
        self.assertAlmostEqual(float(normalized[:, 0].mean()), 0.0, places=6)
        self.assertAlmostEqual(float(normalized[:, -1].mean()), 0.0, places=6)

    def test_feature_indices_stay_stable_when_ports_and_tds_are_enabled(self):
        self.assertEqual(get_port_feature_indices(True), (4, 5))
        self.assertEqual(get_time_delta_feature_indices(True, True), (6, 7))

    def test_create_hetero_obj_swaps_only_port_columns_when_ports_and_tds_are_enabled(self):
        edge_attr = torch.tensor(
            [
                [1.0, 10.0, 100.0, 0.0, 7.0, 8.0, 70.0, 80.0],
                [2.0, 20.0, 200.0, 1.0, 9.0, 10.0, 90.0, 100.0],
            ]
        )
        data = create_hetero_obj(
            x=torch.ones((3, 1)),
            y=torch.tensor([0, 1]),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            edge_attr=edge_attr,
            timestamps=edge_attr[:, 0],
            args=SimpleNamespace(ports=True, tds=True),
        )

        port_indices = list(get_port_feature_indices(True))
        td_indices = list(get_time_delta_feature_indices(True, True))
        expected_reverse = edge_attr.clone()
        expected_reverse[:, port_indices] = expected_reverse[:, list(reversed(port_indices))]

        self.assertTrue(torch.equal(data["node", "to", "node"].edge_attr, edge_attr))
        self.assertTrue(torch.equal(data["node", "rev_to", "node"].edge_attr, expected_reverse))
        self.assertTrue(
            torch.equal(
                data["node", "rev_to", "node"].edge_attr[:, td_indices],
                edge_attr[:, td_indices],
            )
        )

    def test_create_hetero_obj_clones_forward_and_reverse_edge_attributes(self):
        edge_attr = torch.tensor(
            [
                [1.0, 10.0, 100.0, 0.0, 7.0, 8.0],
                [2.0, 20.0, 200.0, 1.0, 9.0, 10.0],
            ]
        )
        data = create_hetero_obj(
            x=torch.ones((3, 1)),
            y=torch.tensor([0, 1]),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            edge_attr=edge_attr,
            timestamps=edge_attr[:, 0],
            args=SimpleNamespace(ports=True, tds=False),
        )

        data["node", "rev_to", "node"].edge_attr[:, 0] = -1

        self.assertTrue(torch.equal(data["node", "to", "node"].edge_attr, edge_attr))


if __name__ == "__main__":
    unittest.main()
