import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from infra.modal.parent_inventory import verify_transfer, SOURCE


class InventoryTests(unittest.TestCase):
    def test_source_schema_creates_admission_manifest_and_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / 'parents'
            receipt_dir = base / 'receipts'
            receipt_dir.mkdir()
            files = []
            for architecture in ('D2W512', 'D4W1536'):
                for env, seeds in {'dog-run': (1, 2, 3), 'humanoid-run': (1, 2, 3),
                                   'myo-key-turn': (1, 2), 'myo-pen-twirl': (1, 2)}.items():
                    for seed in seeds:
                        run = Path(architecture) / env / f'seed_{seed}'
                        state = 'step_000500000_test'
                        metadata = {'identity': {'architecture': architecture, 'environment': env, 'seed': seed},
                                    'code': {'commit': SOURCE}, 'config_hash': 'hash'}
                        payloads = {'DONE': json.dumps({'interaction_step': 500000}),
                                    'run_metadata.json': json.dumps(metadata), 'state/LATEST': state,
                                    **{f'state/{state}/{name}': 'data' for name in
                                       ('meta.pkl', 'buffer.npz', 'buffer_meta.pkl', 'obs_rms.pkl', 'agent_ckpt/checkpoint')}}
                        for name, data in payloads.items():
                            path = root / run / name
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text(data)
                            content = path.read_bytes()
                            files.append({'destination_relative_path': (run / name).as_posix(),
                                          'parent_identity': {'regime': architecture, 'environment': env, 'seed': seed},
                                          'sha256': hashlib.sha256(content).hexdigest(),
                                          'size_bytes': len(content), 'source_commit': SOURCE})
            manifest = {'schema_version': 1, 'source_commit': SOURCE, 'volume': 'exp1-pilot-states',
                        'destination_prefix': 'exp1-pilot-parents', 'file_count': len(files),
                        'total_bytes': sum(f['size_bytes'] for f in files), 'files': files}
            source = receipt_dir / 'manifest.json'
            source.write_text(json.dumps(manifest))
            results = [{'path': f['destination_relative_path'], 'expected_bytes': f['size_bytes'],
                        'actual_bytes': f['size_bytes'], 'expected_sha256': f['sha256'],
                        'actual_sha256': f['sha256'], 'status': 'ok'} for f in files]
            receipt = receipt_dir / 'receipt.json'
            receipt.write_text(json.dumps({'manifest_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                          'volume': 'exp1-pilot-states', 'prefix': 'exp1-pilot-parents',
                                          'source_commit': SOURCE, 'file_count': len(files),
                                          'total_bytes': manifest['total_bytes'], 'verified': True,
                                          'extra': [], 'missing': [], 'results': results}))
            runs = verify_transfer(root, source, receipt)
            self.assertEqual(len(runs), 20)
            self.assertEqual(runs[0]['state'], state)
            (root / files[0]['destination_relative_path']).write_text('changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                verify_transfer(root, source, receipt)
