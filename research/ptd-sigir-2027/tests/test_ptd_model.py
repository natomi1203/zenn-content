from __future__ import annotations

import unittest

try:
    import torch

    from runner.ptd_model import (
        DISTILLATION_ITEM,
        DISTILLATION_NODE,
        LossConfiguration,
        ModelInputs,
        PTDModelConfig,
        TrainingBatch,
        build_registered_model,
        compute_ptd_loss,
        parse_most_recent_first_history,
    )
except ImportError:  # pragma: no cover - default dependency-free test environment
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional production dependency")
class PTDModelContractTest(unittest.TestCase):
    def config(self) -> PTDModelConfig:
        return PTDModelConfig(
            item_hash_bucket_size=64,
            user_hash_bucket_size=32,
            category_hash_bucket_size=16,
        )

    def inputs(self) -> ModelInputs:
        click = torch.zeros((2, 30), dtype=torch.long)
        purchase = torch.zeros((2, 30), dtype=torch.long)
        click[0, :3] = torch.tensor([2, 3, 4])
        purchase[1, :2] = torch.tensor([5, 6])
        return ModelInputs(
            user_ids=torch.tensor([2, 3]),
            click_history_ids=click,
            click_lengths=torch.tensor([3, 0]),
            purchase_history_ids=purchase,
            purchase_lengths=torch.tensor([0, 2]),
            child_item_ids=torch.tensor([[7, 8], [9, 10]]),
            child_category_ids=torch.tensor([[2, 2], [3, 4]]),
            parent_node_ids=torch.tensor([[11, 11], [12, 12]]),
            child_levels=torch.tensor([[13, 13], [4, 4]]),
        )

    def test_registered_history_reversal_and_padding(self) -> None:
        encoded, length = parse_most_recent_first_history(
            "newest;middle;oldest", bucket_size=64
        )
        newest, _ = parse_most_recent_first_history("newest", bucket_size=64)
        oldest, _ = parse_most_recent_first_history("oldest", bucket_size=64)
        self.assertEqual(length, 3)
        self.assertEqual(encoded[0], oldest[0])
        self.assertEqual(encoded[2], newest[0])
        self.assertEqual(encoded[3:], [0] * 27)

    def test_both_encoders_emit_binary_logits_and_ignore_padding_suffix(self) -> None:
        for variant in ("ptd_combined", "ptd_combined_baseline_encoder"):
            torch.manual_seed(16630)
            model, _ = build_registered_model(variant, self.config())
            model.eval()
            inputs = self.inputs()
            altered = inputs.click_history_ids.clone()
            altered[0, 3:] = 20
            altered[1, :] = 21
            changed = ModelInputs(
                **{
                    **inputs.__dict__,
                    "click_history_ids": altered,
                }
            )
            with torch.inference_mode():
                first = model(inputs)
                second = model(changed)
            self.assertEqual(tuple(first.shape), (2, 2))
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_item_and_node_terms_are_independent(self) -> None:
        base = self.inputs()
        selection = torch.zeros(13, dtype=torch.long)
        inputs = ModelInputs(
            user_ids=base.user_ids[selection],
            click_history_ids=base.click_history_ids[selection],
            click_lengths=base.click_lengths[selection],
            purchase_history_ids=base.purchase_history_ids[selection],
            purchase_lengths=base.purchase_lengths[selection],
            child_item_ids=base.child_item_ids[selection],
            child_category_ids=base.child_category_ids[selection],
            parent_node_ids=base.parent_node_ids[selection],
            child_levels=torch.arange(1, 14)[:, None].expand(-1, 2),
        )
        batch = TrainingBatch(
            inputs=inputs,
            path_group_ids=torch.zeros(13, dtype=torch.long),
            positive_child=torch.arange(13) % 2,
            teacher_probabilities=torch.tensor([[0.8, 0.2]] * 13),
            distillation_kind=torch.tensor(
                [DISTILLATION_NODE] * 12 + [DISTILLATION_ITEM]
            ),
        )
        logits = torch.tensor([[0.2, -0.1]] * 13, requires_grad=True)
        item_only = compute_ptd_loss(
            logits,
            batch,
            LossConfiguration(2.0, 0.3, 0.3, True, False),
        )
        node_only = compute_ptd_loss(
            logits,
            batch,
            LossConfiguration(2.0, 0.3, 0.3, False, True),
        )
        self.assertGreater(float(item_only["item_term"].detach()), 0)
        self.assertEqual(float(item_only["node_term"].detach()), 0)
        self.assertEqual(float(node_only["item_term"].detach()), 0)
        self.assertGreater(float(node_only["node_term"].detach()), 0)
        expected_supervised = torch.nn.functional.cross_entropy(
            logits, batch.positive_child, reduction="sum"
        )
        torch.testing.assert_close(item_only["supervised"], expected_supervised)
        torch.testing.assert_close(node_only["node_kl"], item_only["item_kl"] * 12)

    def test_contract_rejects_unregistered_architecture(self) -> None:
        with self.assertRaises(ValueError):
            PTDModelConfig(hidden_dim=32)
        with self.assertRaises(ValueError):
            PTDModelConfig(windows=(1, 29))

    def test_registered_variant_switches_are_explicit(self) -> None:
        expectations = {
            "fixed_tdm": ("hstu", (False, False)),
            "ptd_item": ("hstu", (True, False)),
            "ptd_node": ("hstu", (False, True)),
            "ptd_combined": ("hstu", (True, True)),
            "alternating_tdm": ("hstu", (False, False)),
            "alternating_ptd": ("hstu", (True, True)),
            "ptd_combined_baseline_encoder": ("din", (True, True)),
        }
        for variant, (encoder, switches) in expectations.items():
            model, observed = build_registered_model(variant, self.config())
            self.assertEqual(model.encoder_name, encoder)
            self.assertEqual(observed, switches)
        with self.assertRaises(ValueError):
            build_registered_model("teacher_oracle", self.config())


if __name__ == "__main__":
    unittest.main()
