"""Portable installer unit tests: no downloads, package installation, or large model load.

From the repository root: python -m unittest discover -s tests -v
"""
from pathlib import Path
from unittest.mock import patch
import hashlib
import io
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import prepare_models as models
import prepare_offline as offline
import self_check
import setup_environment as setup


class InstallerTests(unittest.TestCase):
    def test_locked_versions_and_index_modes(self):
        lock = dict(setup.locked_requirements())
        self.assertEqual(len(lock), 190)
        self.assertEqual(lock['torch'], '2.8.0+cpu')
        self.assertEqual(lock['torchvision'], '0.23.0+cpu')
        online = setup.pip_command('python.exe')
        local = setup.pip_command('python.exe', wheelhouse=Path('local-wheels'))
        self.assertIn(setup.PYPI, online)
        self.assertIn(setup.TORCH_INDEX, online)
        self.assertIn('--no-index', local)
        self.assertNotIn('--index-url', local)
        self.assertNotIn('--extra-index-url', local)

    def test_shipped_core_data_hashes(self):
        self.assertEqual(self_check.asset_failures(root=ROOT, core=True, full_hash=True), [])

    def test_model_pins_local_metadata_and_path_boundaries(self):
        plan = models.model_plan(ROOT)
        self.assertEqual(len(plan), 37)
        self.assertEqual(sum(row['local_metadata'] for row in plan), 4)
        for row in plan:
            if row['local_metadata']:
                self.assertIsNone(row['url'])
            elif row['model'] != 'resnet18':
                self.assertIn('/resolve/' + row['revision'] + '/', row['url'])
                self.assertRegex(row['revision'], r'^[0-9a-f]{40}$')
        with self.assertRaises(ValueError):
            models.checked_path(ROOT, '../escape.bin')
        with self.assertRaises(ValueError):
            models.model_plan(ROOT, names=['unknown'])

    def test_download_hash_failure_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'model.bin'
            destination.write_bytes(b'old')
            payload = b'pinned payload'
            row = {'path': 'test', 'bytes': len(payload),
                   'sha256': hashlib.sha256(payload).hexdigest()}
            with self.assertRaises(ValueError):
                models.download_verified('https://example.invalid/pinned', destination, row,
                                         opener=lambda *args, **kwargs: io.BytesIO(b'bad'))
            self.assertEqual(destination.read_bytes(), b'old')
            self.assertEqual(list(Path(directory).glob('*.part')), [])
            models.download_verified('https://example.invalid/pinned', destination, row,
                                     opener=lambda *args, **kwargs: io.BytesIO(payload))
            self.assertTrue(models.matches(destination, row))

    def test_wheel_selection_ignores_nested_vendor_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / 'example-1.0-py3-none-any.whl'
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr('example-1.0.dist-info/METADATA', 'Name: example\nVersion: 1.0\n')
                archive.writestr('example/vendor/other-9.dist-info/METADATA', 'Name: other\nVersion: 9\n')
            self.assertEqual(offline.select_wheels(Path(directory), [('example', '1.0')]), [wheel])
            with self.assertRaises(ValueError):
                offline.select_wheels(Path(directory), [('example', '2.0')])

    def test_runtime_copy_is_repeatable_and_preserves_mismatched_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            source, destination = base / 'source', base / 'runtime/python'
            (source / 'Lib/site-packages').mkdir(parents=True)
            (source / 'Lib/standard.py').write_bytes(b'stdlib')
            (source / 'Lib/site-packages/unused.py').write_bytes(b'external-package')
            (source / 'python.exe').write_bytes(b'test executable')
            self.assertTrue(destination.resolve().is_relative_to(base))
            # Probe calls are mocked because this is a tiny artificial runtime, not
            # an interpreter copied from the contributor's computer.
            with patch.object(offline, 'python_info', return_value={}), patch.object(offline.subprocess, 'run'):
                offline.copy_runtime(source, destination)
                self.assertFalse((destination / 'Lib/site-packages').exists())
                offline.copy_runtime(source, destination)
                (destination / 'Lib/standard.py').write_bytes(b'changed')
                with self.assertRaises(RuntimeError):
                    offline.copy_runtime(source, destination)
                self.assertEqual((destination / 'Lib/standard.py').read_bytes(), b'changed')
                self.assertEqual((source / 'Lib/standard.py').read_bytes(), b'stdlib')

    def test_online_preparation_does_not_inherit_learning_socket_guard(self):
        with patch.dict(setup.os.environ, {'COURSE_OFFLINE': '1', 'HF_HUB_OFFLINE': '1',
                                          'PYTHONPATH': 'old-guard'}):
            online = setup.child_environment()
            local = setup.child_environment(True)
        self.assertNotIn('COURSE_OFFLINE', online)
        self.assertNotIn('PYTHONPATH', online)
        self.assertEqual(local['COURSE_OFFLINE'], '1')
        self.assertIn('offline_guard', local['PYTHONPATH'])


if __name__ == '__main__':
    unittest.main()
