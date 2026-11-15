"""Build refusals of tools/build_binary.py over metadata doubles: the named copyleft or non-commercial distributions
(coqui-tts, TTS, ultralytics, phonemizer, phonemizer-fork, espeakng-loader) and any distribution whose License-Expression,
licence classifier or short License field declares GPL, AGPL or non-commercial terms are refused unless the builder opts
in by name; LGPL, permissive licences and long bundled licence texts are not refused; build_plan raises before collecting
anything and records an opt-in; --allow-restricted is parsed. The functions are lifted from the source like
check_engine_excludes.py does, so no build tool, PyInstaller, model or network is involved."""
import ast
import re
import sys
from email.message import Message
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tools' / 'build_binary.py'
TREE = ast.parse(SOURCE.read_text(encoding='utf-8'))
NAMES = {'RESTRICTED_DISTRIBUTIONS', 'COPYLEFT_LICENSE', 'NONCOMMERCIAL_LICENSE', 'restricted_distributions'}
NODES = [node for node in TREE.body if (isinstance(node, ast.FunctionDef) and node.name in NAMES)
         or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in NAMES for t in node.targets))]
namespace = {'re': re}
exec('from packaging.utils import canonicalize_name', namespace)
exec(compile(ast.Module(body=NODES, type_ignores=[]), str(SOURCE), 'exec'), namespace)
restricted_distributions = namespace['restricted_distributions']


class Distribution:
    def __init__(self, license_expression=None, classifiers=(), license_text=None):
        self.metadata = Message()
        if license_expression:
            self.metadata['License-Expression'] = license_expression
        for value in classifiers:
            self.metadata['Classifier'] = value
        if license_text:
            self.metadata['License'] = license_text


def check_named_and_declared():
    installed = {
        'Coqui-TTS': Distribution('MPL-2.0'), 'ultralytics': Distribution('AGPL-3.0'), 'phonemizer_fork': Distribution(),
        'espeakng-loader': Distribution(), 'TTS': Distribution(license_text='MPL-2.0'),
        'gpl-tool': Distribution(classifiers=['License :: OSI Approved :: GNU General Public License v3 (GPLv3)']),
        'agpl-lib': Distribution('AGPL-3.0-or-later'), 'gpl-short': Distribution(license_text='GPLv3'),
        'nc-weights': Distribution(license_text='CC-BY-NC-4.0'), 'nc-expr': Distribution('LicenseRef-NonCommercial'),
        # not refused: lesser GPL, permissive, a bundled licence text, a dual classifier list without GPL
        'lgpl-lib': Distribution('LGPL-3.0-or-later', ['License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)']),
        'mit-lib': Distribution('MIT', ['License :: OSI Approved :: MIT License']),
        'numpy-like': Distribution(license_text='Copyright (c) 2005 NumPy Developers\n\nThe GNU General Public License '
                                                'with the GCC Runtime Library Exception covers libgfortran...' + 'x' * 200),
        'gpl-substring': Distribution('Apache-2.0', ['License :: OSI Approved :: Apache Software License']),
    }
    refused = restricted_distributions(installed)
    assert set(refused) == {'coqui-tts', 'ultralytics', 'phonemizer-fork', 'espeakng-loader', 'tts', 'gpl-tool', 'agpl-lib',
                            'gpl-short', 'nc-weights', 'nc-expr'}, refused
    assert 'non-commercial Coqui Public Model License' in refused['coqui-tts'] and refused['ultralytics'] == 'AGPL-3.0'
    assert refused['gpl-tool'].startswith('declared licence: License :: OSI Approved :: GNU General Public License')
    # an opt-in by any spelling of the name removes that distribution only
    allowed = restricted_distributions(installed, ['coqui_tts', 'Ultralytics'])
    assert 'coqui-tts' not in allowed and 'ultralytics' not in allowed and 'gpl-tool' in allowed
    assert restricted_distributions({'mit-lib': installed['mit-lib']}) == {}


def check_build_plan_wiring():
    text = SOURCE.read_text(encoding='utf-8')
    plan = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == 'build_plan')
    assert [arg.arg for arg in plan.args.args] == ['profile', 'allow_restricted'], 'build_plan(profile, allow_restricted=())'
    body = ast.get_source_segment(text, plan)
    refusal = body.index('restricted_distributions(installed, allow_restricted)')
    assert refusal < body.index('engine_excludes(installed)'), 'the refusal comes before anything is collected'
    assert "'restricted_opt_in': restricted_opt_in" in body
    assert "'restricted_opt_in': plan.get('restricted_opt_in', [])" in text, 'DEPENDENCIES.json records the opt-in'
    assert "parser.add_argument('--allow-restricted', action='append', default=[]" in text
    assert 'plan = build_plan(args.profile, args.allow_restricted)' in text


def main():
    check_named_and_declared()
    check_build_plan_wiring()
    print('{"check": "build_refusals", "result": "pass", "scope": "metadata doubles and source wiring; no build or network"}')


if __name__ == '__main__':
    main()
