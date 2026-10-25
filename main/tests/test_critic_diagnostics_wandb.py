"""W&B transport tests: scalar/file fixtures only; no training or probe fits."""

import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import wandb

from scale_rl.common.logger import WandbTrainerLogger
from scale_rl.diagnostics.critic_metrics import PREFIX
from scale_rl.diagnostics.wandb_logging import publish_masks, scalar_payload


def observation(root):
    path = root / "interaction_000000020.npz"
    np.savez_compressed(path, mask_0000=np.array([True, False, True]))
    record = {
        "schema": PREFIX,
        "interaction_step": 20,
        "optimizer_step": 17,
        "mask_file": path.name,
        "mask_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "inputs": {"current_replay": {"sha256": "synthetic-input", "examples": 3}},
        "metrics": {
            f"{PREFIX}/current_replay/example_rank": 2.5,
            f"{PREFIX}/current_replay/example_percent": 100 / 3,
            f"{PREFIX}/current_replay/example_undefined_overlap": None,
        },
    }
    path.with_suffix(".json").write_text(json.dumps(record))
    (root / "session.json").write_text(
        json.dumps({"evidence": "synthetic file fixture"})
    )
    np.savez(
        root / "frozen_inputs.npz",
        observation=np.zeros((3, 1)),
        action=np.zeros((3, 1)),
    )
    return record


def logger_fixture():
    logger = WandbTrainerLogger.__new__(WandbTrainerLogger)
    logger.reset()
    logger.update_metric(**{"train/critic_loss": 7.0})
    return logger


class WandbCriticTransportTest(unittest.TestCase):
    def test_single_existing_history_call_keeps_values_counters_and_historical_keys(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            wandb, "run", None
        ):
            record = observation(Path(tmp))
            logger = logger_fixture()
            logger.update_critic_diagnostics(record, tmp, step=40)
            with mock.patch.object(wandb, "log") as log:
                logger.log_metric(step=40)
            log.assert_called_once()
            self.assertEqual(log.call_args.kwargs, {"step": 40})
            payload = log.call_args.args[0]
            self.assertEqual(payload["train/critic_loss"], 7.0)
            for key, value in record["metrics"].items():
                self.assertEqual(payload[key], value)
            self.assertEqual(payload[PREFIX + "/interaction_step"], 20)
            self.assertEqual(payload[PREFIX + "/optimizer_step"], 17)
            self.assertEqual(payload[PREFIX + "/raw_environment_step"], 40)
            logger.reset()
            self.assertFalse(any(k.startswith(PREFIX) for k in logger.media_dict))

    def test_skipped_observation_does_not_repeat_stale_metrics_or_upload(self):
        logger = logger_fixture()
        with mock.patch("scale_rl.diagnostics.wandb_logging.publish_masks") as upload:
            logger.update_critic_diagnostics(None, "/must/not/read", step=40)
            upload.assert_not_called()
        with mock.patch.object(wandb, "log") as log:
            logger.log_metric(step=40)
        self.assertEqual(log.call_args.args[0], {"train/critic_loss": 7.0})

    def test_mask_artifact_contains_receipt_session_and_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = observation(Path(tmp))
            run = mock.Mock(id="synthetic-run", disabled=False)
            with mock.patch.object(wandb, "run", run), mock.patch.object(
                wandb, "Artifact"
            ) as factory:
                publish_masks(record, tmp, environment_step=40)
            self.assertEqual(
                factory.call_args.kwargs["name"], "critic-diagnostics-synthetic-run"
            )
            self.assertEqual(factory.call_args.kwargs["type"], "critic-diagnostics")
            self.assertEqual(
                factory.call_args.kwargs["metadata"]["raw_environment_step"], 40
            )
            names = {
                c.kwargs["name"] for c in factory.return_value.add_file.call_args_list
            }
            self.assertEqual(
                names,
                {
                    "interaction_000000020.npz",
                    "interaction_000000020.json",
                    "session.json",
                    "frozen_inputs.npz",
                },
            )
            run.log_artifact.assert_called_once_with(
                factory.return_value, aliases=["latest", "interaction-20"]
            )

    def test_wandb_disabled_keeps_sidecars_without_artifact_upload(self):
        with mock.patch.object(
            wandb, "run", mock.Mock(disabled=True)
        ), mock.patch.object(wandb, "Artifact") as artifact:
            self.assertIsNone(publish_masks({}, "/must/not/read", environment_step=40))
            artifact.assert_not_called()

    def test_corrupt_mask_or_wrong_namespace_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = observation(Path(tmp))
            record["metrics"]["train/critic_loss"] = 99
            with self.assertRaisesRegex(ValueError, "historical"):
                scalar_payload(record, environment_step=40)
            record["metrics"].pop("train/critic_loss")
            (Path(tmp) / record["mask_file"]).write_bytes(b"corrupt")
            with mock.patch.object(wandb, "run", mock.Mock(id="r", disabled=False)):
                with self.assertRaisesRegex(ValueError, "checksum"):
                    publish_masks(record, tmp, environment_step=40)

    def test_actual_offline_sdk_preserves_history_and_accepts_artifact(self):
        # Actual W&B SDK and local datastore; no authenticated online run/upload.
        from wandb.sdk.internal.datastore import DataStore
        from wandb.proto import wandb_internal_pb2

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"WANDB_SILENT": "true"}
        ):
            root = Path(tmp)
            os.environ["WANDB_DATA_DIR"] = str(root / "staging")
            os.environ["WANDB_CACHE_DIR"] = str(root / "cache")
            os.environ["WANDB_ARTIFACT_DIR"] = str(root / "artifacts")
            record = observation(root)
            run = wandb.init(
                project="echocritic-diagnostic-cpu-test",
                mode="offline",
                dir=tmp,
                settings=wandb.Settings(
                    disable_git=True, disable_code=True, console="off"
                ),
            )
            try:
                logger = logger_fixture()
                before_np, before_py = pickle.dumps(
                    np.random.get_state()
                ), pickle.dumps(random.getstate())
                with mock.patch.object(
                    run, "log_artifact", wraps=run.log_artifact
                ) as upload:
                    logger.update_critic_diagnostics(record, root, step=40)
                artifact = upload.call_args.args[0]
                self.assertEqual(
                    set(artifact.manifest.entries),
                    {
                        "interaction_000000020.npz",
                        "interaction_000000020.json",
                        "session.json",
                        "frozen_inputs.npz",
                    },
                )
                logger.log_metric(step=40)
                self.assertEqual(before_np, pickle.dumps(np.random.get_state()))
                self.assertEqual(before_py, pickle.dumps(random.getstate()))
                data_path = Path(run.dir).parent / f"run-{run.id}.wandb"
            finally:
                run.finish()
            store = DataStore()
            store.open_for_scan(str(data_path))
            rows = []
            while True:
                chunk = store.scan_data()
                if chunk is None:
                    break
                message = wandb_internal_pb2.Record()
                message.ParseFromString(chunk)
                if message.WhichOneof("record_type") == "history":
                    rows.append(
                        {
                            item.key
                            or "/".join(item.nested_key): json.loads(item.value_json)
                            for item in message.history.item
                        }
                    )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["_step"], 40)
            self.assertEqual(rows[0]["train/critic_loss"], 7.0)
            for key, value in record["metrics"].items():
                self.assertEqual(rows[0][key], value)
            self.assertEqual(rows[0][PREFIX + "/interaction_step"], 20)
