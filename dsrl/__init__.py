__version__ = "0.1.0"

import os
import sys
from dsrl.offline_env import OfflineEnvWrapper, OfflineEnv

__all__ = [
    "offline_env",
]

SUPPRESS_MESSAGES = bool(os.environ.get('DSRL_SUPPRESS_IMPORT_ERROR', 0))

try:
    import dsrl.offline_bullet_safety_gym
    import dsrl.offline_safety_gymnasium
    import dsrl.offline_metadrive
except ImportError as e:
    if not SUPPRESS_MESSAGES:
        print('Warning: failed to import.', file=sys.stderr)
        print(e, file=sys.stderr)
