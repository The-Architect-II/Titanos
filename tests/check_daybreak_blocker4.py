"""Blocker 4 (explicit resident, no host materialisation): independent static and validator checks.

No Torch, ACE, model files, devices or network. The runner source is inspected as an AST; the
parent's _verify_runtime is exercised with the fixture receipts from check_music_device.
Exit 0 means every blocker-4 property below holds in the working tree.
"""
import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
RUNNER = ROOT / 'tools/music_runtime/ace_adapter.py'
TREE = ast.parse(RUNNER.read_text(encoding='utf-8'), filename=str(RUNNER))


def attribute_chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return '.'.join(reversed(parts))


def calls(root, suffix):
    """Call nodes whose callee dotted name ends with `suffix` (e.g. 'AutoencoderOobleck.from_pretrained')."""
    for node in ast.walk(root):
        if isinstance(node, ast.Call):
            name = attribute_chain(node.func)
            if name == suffix or name.endswith('.' + suffix):
                yield node


def keyword(call, name):
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


def function(root, name):
    found = [node for node in ast.walk(root) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(found) != 1:
        raise AssertionError('expected exactly one def {}, found {}'.format(name, len(found)))
    return found[0]


class ResidentNoHostMaterialisation(unittest.TestCase):
    def test_vae_loads_straight_to_the_device_and_is_never_moved_from_the_host(self):
        loader = function(TREE, 'load_components')
        pretrained = list(calls(loader, 'AutoencoderOobleck.from_pretrained'))
        device_map_route = any(keyword(call, 'device_map') is not None for call in pretrained)
        # Alternative: meta-init, safetensors load_file(device=...) then load_state_dict(assign=True).
        load_file_to_device = any(keyword(call, 'device') is not None for call in calls(loader, 'load_file'))
        assign_load = any(isinstance(keyword(call, 'assign'), ast.Constant) and keyword(call, 'assign').value is True
                          for call in calls(loader, 'load_state_dict'))
        # Current route: the VAE is built through build_resident (meta-init + stream_safetensors per tensor -> load_state_dict(assign=True)),
        # with no host mapping at all; accept it when load_components constructs the VAE via build_resident and build_resident assigns.
        builder = function(TREE, 'build_resident')
        streamed_route = (any(True for _ in calls(loader, 'build_resident')) and any(True for _ in calls(builder, 'stream_safetensors'))
                          and any(isinstance(keyword(call, 'assign'), ast.Constant) and keyword(call, 'assign').value is True
                                  for call in calls(builder, 'load_state_dict')))
        self.assertTrue(device_map_route or (load_file_to_device and assign_load) or streamed_route,
                        'resident VAE must load onto the device (diffusers device_map, or meta-init + load_file(device) + '
                        'load_state_dict(assign=True)); today AutoencoderOobleck.from_pretrained has no device_map at '
                        'ace_adapter.py:{}'.format(pretrained[0].lineno if pretrained else '?'))
        moves = [call for call in calls(loader, 'to') if attribute_chain(call.func) == 'self.vae.to']
        self.assertEqual([call.lineno for call in moves], [],
                         'load_components must not move self.vae after loading (a post-load .to(device) means the weights '
                         'materialised on the host first)')

    def test_silence_latent_is_loaded_with_the_device_as_map_location(self):
        loads = list(calls(TREE, 'torch.load'))
        self.assertTrue(loads, 'runner must load the pinned silence tensor with torch.load')
        for call in loads:
            target = keyword(call, 'map_location')
            self.assertIsNotNone(target, 'torch.load without map_location at line {}'.format(call.lineno))
            self.assertFalse(isinstance(target, ast.Constant) and target.value == 'cpu',
                             "torch.load(map_location='cpu') at ace_adapter.py:{} materialises silence_latent on the host; "
                             'resident mode requires map_location=device'.format(call.lineno))

    def test_component_context_never_moves_to_cpu_when_resident(self):
        manager = function(TREE, 'component_manager')
        # Every model.to('cpu') in the manager must sit under a `resident` false branch (else of `if resident`).
        for node in ast.walk(manager):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'resident':
                body_moves = [c for stmt in node.body for c in calls(stmt, 'to')
                              if any(isinstance(a, ast.Constant) and a.value == 'cpu' for a in c.args)]
                self.assertEqual(body_moves, [], "resident branch of component_context moves a component to 'cpu'")


class ParentValidatorRejectsNonResidentUnderResident(unittest.TestCase):
    def setUp(self):
        import check_music_device as fixtures
        from titanos import music_backend
        from titanos.engine import EngineError
        self.fixtures, self.music, self.EngineError = fixtures, music_backend, EngineError

    def test_host_resolution_under_explicit_resident_is_rejected_on_cpu(self):
        contract = self.fixtures.contract('cpu', placement='resident')
        self.assertEqual(contract['placement'], 'resident')
        receipt = self.fixtures.response(contract, resolved='host')
        with self.assertRaises(self.EngineError, msg="_verify_runtime accepted resolved 'host' under contract 'resident' "
                               '(music_backend.py: the resident/offload agreement check is gated on device!="cpu")'):
            self.music._verify_runtime(receipt, contract)

    def test_offload_and_host_resolutions_under_explicit_resident_are_rejected_on_cuda(self):
        contract = self.fixtures.contract('cuda:0', placement='resident')
        for resolved in ('offload', 'host'):
            with self.subTest(resolved=resolved), self.assertRaises(self.EngineError):
                self.music._verify_runtime(self.fixtures.response(contract, 'cuda:0', self.fixtures.B, resolved=resolved), contract)

    def test_resident_receipt_with_matching_evidence_is_still_accepted(self):
        contract = self.fixtures.contract('cuda:0', placement='resident')
        receipt = self.fixtures.response(contract, 'cuda:0', self.fixtures.B, resolved='resident')
        self.assertEqual(self.music._verify_runtime(receipt, contract)['placement']['resolved'], 'resident')


if __name__ == '__main__':
    unittest.main()
