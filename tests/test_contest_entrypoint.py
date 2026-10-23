from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

entry = load_module('contest_entry', 'run.py')
merge = load_module('contest_merge', 'scripts/merge_lora_v3.py')
package = load_module('contest_package', 'scripts/package_submission.py')


class ContestEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / 'base'
        self.base.mkdir()
        (self.base / 'config.json').write_text('{}')
        (self.base / 'model.safetensors').write_bytes(b'fixture')
        self.adapter = self.root / 'adapter'
        self.adapter.mkdir()
        self.config = dict(peft_type='LORA', r=16, lora_alpha=32, bias='none')
        (self.adapter / 'adapter_config.json').write_text(json.dumps(self.config))
        (self.adapter / 'adapter_model.safetensors').write_bytes(b'fixture')
        self.sample = self.root / 'input'
        self.sample.mkdir()
        (self.sample / 'sample.doc').write_bytes(b'fixture')
        self.args = entry.build_parser().parse_args([
            '--input-dir', str(self.sample), '--model', str(self.root / 'merged'),
            '--base-model', str(self.base), '--adapter', str(self.adapter),
            '--result-dir', str(self.root / 'results'), '--work-dir', str(self.root / 'work'),
        ])

    def preflight(self):
        with patch('importlib.util.find_spec', return_value=object()):
            return entry.preflight(self.args, list(self.sample.iterdir()))

    def test_defaults_anchor_to_submission_not_working_directory(self):
        args = entry.build_parser().parse_args([])
        self.assertEqual(args.result_dir, Path('/workspace/result'))
        self.assertTrue(str(args.work_dir).startswith('/workspace/'))
        self.assertEqual(args.adapter, Path('/path/to/adapter'))
        merge_args = merge.build_parser().parse_args([
            '--base-model', str(args.base_model), '--adapter', str(args.adapter), '--output', str(self.root / 'merged')])
        self.assertEqual(args.adapter, merge_args.adapter)
        self.assertEqual(args.base_model, Path('/path/to/base-model'))

    def test_default_input_selects_online_test_set_and_excludes_training(self):
        mount = self.root / 'dataset'
        test = mount / '决赛数据集1/赛题1/线上测试集'
        training = mount / '决赛数据集1/赛题1/训练集/标签'
        test.mkdir(parents=True)
        training.mkdir(parents=True)
        (test / 'test.doc').touch()
        (training / 'label.docx').touch()
        self.assertEqual(entry.select_input_dir(mount), test)

    def test_default_input_refuses_ambiguous_test_directories(self):
        mount = self.root / 'dataset'
        for name in ['first', 'second']:
            (mount / name / '线上测试集').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, '唯一定位'):
            entry.select_input_dir(mount)

    def test_default_input_accepts_flat_reports_but_not_unknown_nested_data(self):
        self.assertEqual(entry.select_input_dir(self.sample), self.sample)
        (self.sample / '训练标签').mkdir()
        (self.sample / '训练标签/label.docx').touch()
        with self.assertRaisesRegex(ValueError, '唯一定位'):
            entry.select_input_dir(self.sample)

    def test_missing_doc_converter_fails_before_model_loading(self):
        with patch('shutil.which', return_value=None):
            self.assertTrue(any('soffice' in e for e in self.preflight()))
            text = self.root / 'text'
            text.mkdir()
            (text / 'sample.txt').write_text('sample')
            self.args.text_dir = text
            errors = self.preflight()
            self.assertTrue(any('soffice' in e for e in errors))
            self.assertTrue(any('来源绑定' in e for e in errors))

    def test_default_sdk_engine_requires_sdk_dependency(self):
        self.assertEqual(self.args.engine, 'tongagents')
        with patch('importlib.util.find_spec', side_effect=lambda name: None if name == 'tongagents' else object()), \
                patch('shutil.which', return_value='/usr/bin/soffice'):
            errors = entry.preflight(self.args, list(self.sample.iterdir()))
        self.assertIn('缺少 Python 依赖：tongagents', errors)
        self.assertTrue(any('模型未准备' in e for e in errors))

    def test_raw_base_cannot_replace_finetuned_model(self):
        self.args.model = str(self.base)
        with patch('shutil.which', return_value='/usr/bin/soffice'):
            self.assertTrue(any('遗漏微调权重' in e for e in self.preflight()))

    def test_existing_incomplete_cache_is_rejected(self):
        model = Path(self.args.model)
        model.mkdir()
        (model / '.merge_incomplete').touch()
        with patch('shutil.which', return_value='/usr/bin/soffice'):
            self.assertTrue(any('尚未保存完成' in e for e in self.preflight()))

    def test_missing_cache_requires_explicit_preparation(self):
        with patch('subprocess.run') as run:
            with self.assertRaises(FileNotFoundError):
                entry.prepare_model(self.args)
        run.assert_not_called()
        Path(self.args.model).mkdir()
        with patch('subprocess.run') as run:
            entry.prepare_model(self.args)
        run.assert_not_called()

    def test_merge_rejects_overwrite_and_unsupported_weight_semantics(self):
        args = SimpleNamespace(base_model=self.base, adapter=self.adapter, output=self.root / 'merged')
        self.assertEqual(merge.validate_paths(args)['lora_alpha'], 32)
        self.config['use_rslora'] = True
        (self.adapter / 'adapter_config.json').write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, '普通 LoRA'):
            merge.validate_paths(args)
        args.output = self.base
        with self.assertRaisesRegex(ValueError, '重叠'):
            merge.validate_paths(args)
        args.output = Path('/model/bridge_merged')
        with self.assertRaisesRegex(ValueError, '只读'):
            merge.validate_paths(args)

    def test_package_includes_only_runtime_files(self):
        source = self.root / 'source'
        source.mkdir()
        for name in package.RUNTIME_FILES:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        for name in ['run_contest.py', 'models/adapter/adapter_model.safetensors',
                     'src/bridge_evidence/train.py', 'scripts/train.py',
                     'data/train.json', 'work/old.tar.gz', 'tests/test_old.py',
                     'src/bridge_evidence/contest/__pycache__/old.pyc']:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('must not be packaged')
        out = self.root / 'package/code'
        package.copy_code(source, out)
        self.assertEqual({p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()},
                         set(package.RUNTIME_FILES))
        for name in package.RUNTIME_FILES:
            self.assertEqual((out / name).read_text(), name)


if __name__ == '__main__':
    unittest.main()
