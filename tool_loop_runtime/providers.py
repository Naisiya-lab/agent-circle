"""Provider configuration shared by direct inference and all agent roles."""
from copy import deepcopy

ROLES = ('understand', 'select', 'solve', 'check')
PROTECTED = {'messages', 'model', 'stream', 'max_tokens', 'max_completion_tokens'}


def profile(cfg, role):
    return cfg.get('role_apis', {}).get(role)


def model_options(cfg, role):
    item = profile(cfg, role)
    if item is None:
        return None  # compatibility with the preserved GLM/Qwen configuration
    options = deepcopy(item.get('request_options', {}))
    if PROTECTED.intersection(options):
        raise ValueError('request_options cannot override model, messages, stream or output limit')
    return options


def build_request(cfg, role, messages):
    item = profile(cfg, role)
    if item is None:
        raise ValueError('The portable direct baseline requires an explicit role_apis profile')
    field = item.get('output_limit_parameter', 'max_tokens')
    if field not in ('max_tokens', 'max_completion_tokens'):
        raise ValueError('Unsupported output limit parameter')
    return {'model': cfg['models'][role], 'stream': cfg.get('stream', True),
            field: cfg['max_tokens'][role], 'messages': messages,
            **model_options(cfg, role)}


def validate_homogeneous(cfg):
    """Enforce M versus a circle consisting only of the same M checkpoint."""
    if cfg.get('experiment', {}).get('homogeneous', True):
        if len(set(cfg['models'][r] for r in ROLES)) != 1:
            raise ValueError('Homogeneous comparison requires exactly the same model in every role')
        signatures = {(cfg['role_apis'][r]['endpoint'], cfg['role_apis'][r].get('model_revision', '')) for r in ROLES}
        if len(signatures) != 1:
            raise ValueError('Homogeneous comparison requires the same endpoint and model revision')
