import logging
import os
import sys
from pathlib import Path


def run(root=None, *, login_only=False):
    root = Path(root or Path(__file__).parent).resolve()
    sys.path.insert(0, str(root / 'src'))
    os.environ.setdefault('LRAEP_CACHE_DIR', str(root / '.runtime/cache'))
    os.environ.setdefault('LRAEP_DATA_DIR', str(root / '.runtime'))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', stream=sys.stdout, force=True)
    try:
        from LRAEP_trainer.config import load_config, resolve_pool_settings
        from LRAEP_trainer.network import routed
        from LRAEP_trainer.pool_training import run_pool_training, validate_pool_endpoint, validate_pool_identity
        from LRAEP_trainer.protocol import load_official_helpers
        from LRAEP_trainer.vpn import ManagedVPN
        config = load_config(root / '.config')
        config['pool'] = resolve_pool_settings(config['pool'])
        if config['pool']['protocol'] != 'WR':
            raise ValueError('WR is required')
        if not config['vpn']['enabled'] or not config['vpn']['link']:
            raise ValueError('An enabled VPN link is required')
        validate_pool_endpoint(config['pool'])
        validate_pool_identity(config['pool'], validate_address=False)
        load_official_helpers(config['official_source'])
        if not login_only:
            import torch
            if torch.device(config['device']).type != 'cuda' or not torch.cuda.is_available():
                raise ValueError('CUDA is required')
        with ManagedVPN(config['vpn']) as proxy, routed(proxy):
            print('WR training | VPN enabled', flush=True)
            return run_pool_training(config, login_only=login_only)
    except KeyboardInterrupt:
        print('Stopped', flush=True)
        return 130
    except (ValueError, RuntimeError, ImportError, OSError) as error:
        print(f'Training failed ({type(error).__name__}): {error}', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(run())
