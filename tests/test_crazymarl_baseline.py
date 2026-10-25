import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct

import pytest

from uav_transport import crazymarl_baseline as baseline


def setup_module_object():
    path=Path(__file__).resolve().parents[1]/'scripts/setup_crazymarl.py'
    spec=importlib.util.spec_from_file_location('crazymarl_setup_for_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def synthetic_elf():
    blob=bytearray(120);blob[:6]=b'\x7fELF\x02\x01'
    struct.pack_into('<Q',blob,32,64)
    struct.pack_into('<H',blob,54,56);struct.pack_into('<H',blob,56,1)
    struct.pack_into('<II',blob,64,0x6474e551,7)
    return bytes(blob)


def test_execstack_repair_changes_only_expected_flag_and_does_not_mutate_uv_cache_hardlinks(tmp_path,monkeypatch):
    module=setup_module_object();original=synthetic_elf();repaired=bytearray(original)
    struct.pack_into('<I',repaired,68,6)
    monkeypatch.setattr(module,'XLA_ORIGINAL_SHA256',hashlib.sha256(original).hexdigest())
    monkeypatch.setattr(module,'XLA_REPAIRED_SHA256',hashlib.sha256(repaired).hexdigest())
    library=tmp_path/'installed.so';library.write_bytes(original)
    cache=tmp_path/'cached.so';os.link(library,cache)
    record=module.repair_execstack(library,tmp_path/'provenance.json')
    assert library.read_bytes()==repaired and cache.read_bytes()==original
    assert record['program_headers']==[dict(index=0,old_flags=7,new_flags=6)]
    assert record['repair'].startswith('Clear only')
    again=module.repair_execstack(library,tmp_path/'provenance.json')
    assert again['found_sha256']==module.XLA_REPAIRED_SHA256
    assert library.read_bytes()==repaired


def test_execstack_repair_refuses_unknown_binaries_without_writing(tmp_path):
    module=setup_module_object();path=tmp_path/'unknown.so';path.write_bytes(b'unknown binary')
    with pytest.raises(ValueError,match='unrecognized'):
        module.repair_execstack(path,tmp_path/'provenance.json')
    assert path.read_bytes()==b'unknown binary' and not (tmp_path/'provenance.json').exists()


def write_algorithm(tmp_path,text):
    path=tmp_path/'baselines/IPPO/ippo_ff_mabrax.py';path.parent.mkdir(parents=True)
    path.write_text(text);return path


def test_adapter_omits_only_unused_export_import_and_preserves_algorithm_body(tmp_path):
    text='from jax2onnx import to_onnx, onnx_function\n\ndef make_train(config, rng):\n    return config["updates"] + rng\n'
    path=write_algorithm(tmp_path,text)
    module,record=baseline.load_upstream_ippo(tmp_path)
    assert module.make_train({'updates':3},2)==5
    function=next(node for node in ast.parse(text).body if isinstance(node,ast.FunctionDef))
    assert record['algorithm_bodies_sha256']['make_train']==hashlib.sha256(ast.get_source_segment(text,function).encode()).hexdigest()
    assert record['source_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.read_text()==text


def test_adapter_refuses_export_dependency_when_algorithm_actually_uses_it(tmp_path):
    write_algorithm(tmp_path,'from jax2onnx import to_onnx, onnx_function\n\ndef make_train(x):\n    return to_onnx(x)\n')
    with pytest.raises(ValueError,match='is used'):
        baseline.load_upstream_ippo(tmp_path)


def test_upstream_context_restores_working_directory_and_import_path(tmp_path):
    import sys
    cwd=os.getcwd();path=sys.path.copy()
    with pytest.raises(RuntimeError):
        with baseline.upstream_context(tmp_path):
            assert os.getcwd()==str(tmp_path) and sys.path[0]==str(tmp_path)
            raise RuntimeError('exercise cleanup')
    assert os.getcwd()==cwd and sys.path==path


def test_official_training_configuration_stays_separate_from_smoke_overrides(tmp_path):
    pytest.importorskip('yaml')
    defaults=tmp_path/'crazymarl/configs/defaults/training.yaml';defaults.parent.mkdir(parents=True)
    defaults.write_text('training:\n  TOTAL_TIMESTEPS: 1000000000\n  NUM_ENVS: 16384\n  ENV_KWARGS:\n    num_quads: 2\n    payload_mass: 0.01\n')
    three=tmp_path/'crazymarl/train/configs/three_quad.yaml';three.parent.mkdir(parents=True)
    three.write_text('training:\n  TOTAL_TIMESTEPS: 2000000000\n  ENV_KWARGS:\n    num_quads: 3\n')
    full=baseline.official_config(tmp_path)
    assert full['TOTAL_TIMESTEPS']==2000000000 and full['NUM_ENVS']==16384
    assert full['ENV_KWARGS']=={'num_quads':3,'payload_mass':.01}
    full['NUM_ENVS']=2
    assert baseline.official_config(tmp_path)['NUM_ENVS']==16384
