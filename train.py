import atexit
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


_process = None
_output_thread = None
_last_returncode = None
_process_lock = threading.RLock()




def _signal_worker(process, *, force=False):
    try:
        if os.name == 'posix':

            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
        elif force:
            process.kill()
        else:
            process.terminate()
    except ProcessLookupError:
        pass


def _forward_output(process, stream):
    try:
        for line in process.stdout:
            if stream is not None:
                try:
                    stream.write(line)
                    stream.flush()
                except (OSError, ValueError):

                    stream = None
    finally:
        process.stdout.close()


def _watch_worker(process, stream, on_exit=None):
    output = threading.Thread(target=_forward_output, args=(process, stream), daemon=True)
    output.start()
    process.wait()

    if os.name == 'posix':
        _signal_worker(process, force=True)
    output.join()
    if on_exit is not None:
        try:
            on_exit(process.returncode)
        except (OSError, ValueError):
            pass


def _watch_parent(parent_pid):

    while os.getppid() == parent_pid:
        time.sleep(0.5)
    os.killpg(os.getpgrp(), signal.SIGTERM)
    time.sleep(10)
    os.killpg(os.getpgrp(), signal.SIGKILL)


def start(root=None, *, login_only=False, output=None):





    global _process, _output_thread, _last_returncode
    root = Path(root or Path(__file__).parent).resolve()
    with _process_lock:
        if _process is not None and _process.poll() is None:
            raise RuntimeError('Training is already running; call stop_training() first')
        stop_training()
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
                   '--root', str(root), '--parent-pid', str(os.getpid())]
        if login_only:
            command.append('--login-only')
        _process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                                    start_new_session=os.name == 'posix')
        _last_returncode = None
        stream = sys.stdout if output is None else output
        on_exit = getattr(output, 'finish', None)
        _output_thread = threading.Thread(target=_watch_worker, args=(_process, stream, on_exit),
                                          name='WR-training-monitor', daemon=True)
        _output_thread.start()
        return _process


def status():

    with _process_lock:
        return _last_returncode if _process is None else _process.poll()


def stop_training(timeout=10):






    global _process, _output_thread, _last_returncode
    if not isinstance(timeout, (int, float)) or not 0 <= timeout < float('inf'):
        raise ValueError('timeout must be a finite non-negative number')
    with _process_lock:
        process = _process
        if process is None:
            return None
        if process.poll() is None:
            _signal_worker(process)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _signal_worker(process, force=True)
                process.wait(timeout=5)
        if _output_thread is not None:
            _output_thread.join(timeout=5)
        code = process.returncode
        _last_returncode = code
        _process, _output_thread = None, None
        return code


atexit.register(stop_training)


def run(root=None, *, login_only=False):

    root = Path(root or Path(__file__).parent).resolve()
    sys.path.insert(0, str(root / 'src'))
    generate_artifacts = True
    previous_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        from workrunner.paths import set_checkout_directories
        set_checkout_directories(root)
        from workrunner.config import resolve_S3_settings
        from workrunner.login_config import load_training_config
        from workrunner.training_artifacts import training_output
        config_path = root / 'config.json'
        config = load_training_config(config_path)
        generate_artifacts = config['training']['generate_artifacts']
        if not login_only and config['training']['job_sequence'][0]['max_gpu_utilization'] == -1:
            return 0
        sys.dont_write_bytecode = previous_bytecode
        with training_output(generate_artifacts):
            if generate_artifacts:
                logging.basicConfig(level=logging.ERROR, format='%(asctime)s %(message)s', stream=sys.stdout, force=True)
            from workrunner.socks_proxy import routed
            from workrunner.S3_training import run_S3_training, validate_S3_endpoint
            from workrunner.vpn import ManagedVPN
            config['S3'] = resolve_S3_settings(config['S3'])
            if config['S3']['protocol'] != 'WR':
                raise ValueError('WR is required')
            if not config['vpn']['enabled'] or not config['vpn']['link']:
                raise ValueError('An enabled VPN link is required')
            validate_S3_endpoint(config['S3'])
            with ManagedVPN(config['vpn']) as proxy, routed(proxy):
                return run_S3_training(config, login_only=login_only, login_config=config_path)
    except KeyboardInterrupt:
        if generate_artifacts:
            print('Stopped', flush=True)
        return 130
    except (ValueError, RuntimeError, ImportError, OSError) as error:
        if generate_artifacts:
            print(f'Training failed ({type(error).__name__}): {error}', flush=True)
        return 1
    finally:
        sys.dont_write_bytecode = previous_bytecode


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run WR training')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--parent-pid', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--root', type=Path, default=Path(__file__).parent)
    parser.add_argument('--login-only', action='store_true')
    arguments = parser.parse_args()
    if arguments.worker and arguments.parent_pid is not None and os.name == 'posix':
        threading.Thread(target=_watch_parent, args=(arguments.parent_pid,), daemon=True).start()
    raise SystemExit(run(arguments.root, login_only=arguments.login_only))
