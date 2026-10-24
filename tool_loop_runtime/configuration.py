"""Resolve portable configuration; host-specific paths belong in shell variables."""
import json
import os
from copy import deepcopy
from pathlib import Path
from .providers import ROLES, validate_homogeneous

BASE = Path(__file__).resolve().parents[1]


def load_config(model_profile, route='B', runtime=None):
    cfg = json.loads(Path(runtime or BASE/'configs/runtime.json').read_text(encoding='utf-8'))
    m = json.loads(Path(model_profile).read_text(encoding='utf-8'))
    endpoint = os.environ.get('MODEL_ENDPOINT', m.get('endpoint', '')).rstrip('/')
    if not endpoint:
        raise ValueError('Set MODEL_ENDPOINT in scripts/env.local.sh or the model shell profile')
    if not endpoint.endswith('/chat/completions'):
        endpoint += '/chat/completions'
    model = os.environ.get('SERVED_MODEL_NAME', m['model_id'])
    options = deepcopy(m.get('request_options', {}))
    cfg['models'] = {r: model for r in ROLES}
    cfg['role_apis'] = {r: {'endpoint': endpoint, 'key_env': m.get('api_key_env', 'MODEL_API_KEY'),
                           'require_key': m.get('require_key', True), 'request_options': deepcopy(options),
                           'model_revision': os.environ.get('MODEL_REVISION', m.get('revision', '')),
                           'output_limit_parameter': m.get('output_limit_parameter', 'max_tokens')} for r in ROLES}
    cfg['max_tokens'].update(m.get('max_tokens', {}))
    cfg['experiment'] = {'homogeneous': True, 'route': route, 'profile': m['model_id'],
                         'nominal_size_b': m.get('nominal_size_b'),
                         'model_revision': os.environ.get('MODEL_REVISION', m.get('revision', '')),
                         'image_policy': 'No client resize or image alteration; provider preprocessing still differs.'}
    cfg['selector_input'] = 'image_and_html' if route == 'B' else 'text'
    cfg['prompt_overrides'] = ({'understand': 'tool_loop_runtime/prompts/01_understand_nocolor.txt',
                                'select': 'tool_loop_runtime/prompts/02_select_vision.txt'} if route == 'B' else {})
    cfg['network']['proxy_mode'] = os.environ.get('API_PROXY_MODE', 'direct')
    if os.environ.get('GENERATION_TEMPERATURE'):
        for r in ROLES: cfg['role_apis'][r]['request_options']['temperature'] = float(os.environ['GENERATION_TEMPERATURE'])
    validate_homogeneous(cfg)
    return cfg
