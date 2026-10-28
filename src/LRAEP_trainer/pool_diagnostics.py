from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import threading
import uuid
LOG = logging.getLogger('LRAEP_trainer')

class UnknownRequestRecorder:

    def __init__(self, directory, settings):
        self.directory = Path(directory).expanduser()
        self.host = settings['host']
        self.port = settings['port']
        self.tls = settings.get('tls', False)
        self.stop = settings.get('stop_on_unknown_request', True)
        self.last_path = None
        self._lock = threading.Lock()

    def __call__(self, frame, message, reason, *, fatal=True):
        now = datetime.now(timezone.utc)
        record = {'time_utc': now.isoformat(timespec='milliseconds'), 'event': 'unknown-pool-request', 'direction': 'pool-to-miner', 'pool': {'host': self.host, 'port': self.port, 'tls': self.tls}, 'reason': reason, 'raw_frame_hex': bytes(frame).hex(), 'decoded_message': message, 'training_stopped': fatal}
        try:
            with self._lock:
                self.directory.mkdir(parents=True, mode=448, exist_ok=True)
                path = self.directory / f"unknown-{now.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:8]}.json"
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 384)
                with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                    json.dump(record, stream, ensure_ascii=True, indent=2, allow_nan=False)
                    stream.write('\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                self.last_path = path
        except (OSError, ValueError, TypeError):
            LOG.critical('Unrecognized PearlHash request; training stopped. Could not save its diagnostic record to %s', self.directory)
            raise RuntimeError('Could not save the unknown PearlHash request; training stopped') from None
        LOG.log(logging.CRITICAL if fatal else logging.WARNING, 'Unrecognized PearlHash request; %s. Evidence saved to %s', 'training stopped' if fatal else 'request recorded', path)
        if fatal and sys.stderr.isatty():
            sys.stderr.write('\x07')
            sys.stderr.flush()
        return path
