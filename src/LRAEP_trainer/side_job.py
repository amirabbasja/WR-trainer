from __future__ import annotations
import argparse
from contextlib import ExitStack
import copy
import logging
from pathlib import Path
import os
import pickle
import shutil
import tempfile
import torch
from torch import nn
LOG = logging.getLogger('LRAEP_trainer.side_job')
PROJECT_ROOT = Path(__file__).resolve().parents[2]

class _BatchNorm(nn.BatchNorm2d):

    def forward(self, values):
        if self.training and values.numel() // values.shape[1] == 1:
            return nn.functional.batch_norm(values, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)
        return super().forward(values)

class ResidualBlock(nn.Module):

    def __init__(self, inputs, outputs, stride=1):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1, bias=False), _BatchNorm(outputs), nn.ReLU(inplace=True), nn.Conv2d(outputs, outputs, 3, padding=1, bias=False), _BatchNorm(outputs))
        self.skip = nn.Identity() if inputs == outputs and stride == 1 else nn.Sequential(nn.Conv2d(inputs, outputs, 1, stride=stride, bias=False), _BatchNorm(outputs))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, images):
        return self.activation(self.layers(images) + self.skip(images))

class OrientationCNN(nn.Module):

    def __init__(self, channels=(64, 128, 256, 512), blocks_per_stage=(2, 2, 2, 2), hidden_size=512):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, channels[0], 7, stride=2, padding=3, bias=False), _BatchNorm(channels[0]), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2, padding=1))
        layers = []
        inputs = channels[0]
        for stage, (outputs, count) in enumerate(zip(channels, blocks_per_stage, strict=True)):
            for block in range(count):
                layers.append(ResidualBlock(inputs, outputs, 2 if stage and block == 0 else 1))
                inputs = outputs
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels[-1], hidden_size), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(hidden_size, 2))

    def forward(self, images):
        return self.classifier(self.features(self.stem(images)))

def _atomic_copy(source, destination):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.checkpoint-', delete=False) as stream:
            temporary = Path(stream.name)
            with source.open('rb') as previous:
                shutil.copyfileobj(previous, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

def save_checkpoint(directory, checkpoint, *, rotate=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / 'weights.pth'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix='.checkpoint-', delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if rotate and weights.exists():
            _atomic_copy(weights, directory / 'weights_backup.pth')
        _atomic_copy(temporary, directory / 'backup.pth')
        os.replace(temporary, weights)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

class SideJob:

    def __init__(self, config, *, project_root=None, before_log=None):
        self.config = copy.deepcopy(config['side_job'])
        self.device = torch.device(config['device'])
        self.directory = Path(project_root or PROJECT_ROOT) / 'data' / 'side-job'
        self.directory.mkdir(parents=True, exist_ok=True)
        self._stack = ExitStack()
        self.model = None
        self.optimizer = None
        self.scaler = None
        self.source = None
        self.shards = []
        self.iterator = None
        self.pending_batch = None
        self.shard_path = None
        self.epoch = 0
        self.next_shard = None
        self.examples = self.correct = 0
        self.total_loss = 0.0
        self.epoch_batches = 0
        self.total_examples = None
        self.total_batches = None
        self._progress_epoch = 1
        self._progress_phase = 'Preparing'
        self.before_log = before_log
        self.closed = False

    def _log(self, message, *args):
        if self.before_log is not None:
            self.before_log()
        LOG.info(message, *args)

    def status(self, gpu_percent=None):
        epoch = str(self._progress_epoch)
        if self.config['epochs']:
            epoch += f"/{self.config['epochs']}"
        details = ['Side-job', f'Epoch {epoch}']
        if self.total_examples is None:
            details.append(self._progress_phase)
        else:
            percentage = 100 * self.examples / max(self.total_examples, 1)
            details.append(f'Step {self.epoch_batches}/{self.total_batches} ({percentage:.1f}%)')
            if self.examples:
                details.append(f'loss={self.total_loss / self.examples:.4f} accuracy={100 * self.correct / self.examples:.1f}%')
            if self._progress_phase == 'Complete':
                details.append('Complete')
        details.append('GPU N/A' if gpu_percent is None else f'GPU {gpu_percent}%')
        return ' | '.join(details)

    def _architecture(self):
        return {name: self.config[name] for name in ('channels', 'blocks_per_stage', 'hidden_size')}

    def _checkpoint(self):
        return {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'scaler': self.scaler.state_dict(), 'epoch': self.epoch, 'next_shard': self.next_shard, 'architecture': self._architecture()}

    def _initialize(self):
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('side-job requires an available CUDA GPU')
        if self.device.index is not None and self.device.index >= torch.cuda.device_count():
            raise ValueError('side-job device selects an unavailable CUDA GPU')
        from .side_job_data import DatasetSource
        torch.manual_seed(self.config['seed'])
        self.model = OrientationCNN(**self._architecture()).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.config['use_amp'])
        weights = self.directory / 'weights.pth'
        if weights.exists():
            try:
                state = torch.load(weights, map_location='cpu', weights_only=True)
                if isinstance(state, dict) and 'model' in state:
                    if state.get('architecture', self._architecture()) != self._architecture():
                        raise ValueError('architecture mismatch')
                    self.model.load_state_dict(state['model'])
                    if 'optimizer' in state:
                        self.optimizer.load_state_dict(state['optimizer'])
                    if 'scaler' in state:
                        self.scaler.load_state_dict(state['scaler'])
                    self.epoch = state.get('epoch', 0)
                    self.next_shard = state.get('next_shard')
                    if type(self.epoch) is not int or self.epoch < 0:
                        raise ValueError('invalid saved epoch')
                    if self.next_shard is not None and (not isinstance(self.next_shard, str)):
                        raise ValueError('invalid saved shard')
                else:
                    self.model.load_state_dict(state)
            except (OSError, RuntimeError, ValueError, TypeError, KeyError, EOFError, pickle.UnpicklingError) as exc:
                raise RuntimeError('could not load side-job weights.pth; check its architecture and checkpoint format') from exc
            for group in self.optimizer.param_groups:
                group['lr'] = self.config['learning_rate']
                group['weight_decay'] = self.config['weight_decay']
            self._log('Resumed side-job weights at epoch %d', self.epoch)
        else:
            save_checkpoint(self.directory, self._checkpoint(), rotate=False)
            self._log('Created new side-job weights')
        self._progress_epoch = self.epoch + 1
        self._log('Side-job CNN: %s; %s parameters', torch.cuda.get_device_name(self.device), f'{sum((parameter.numel() for parameter in self.model.parameters())):,}')
        if self.config['epochs'] and self.epoch >= self.config['epochs']:
            self._progress_epoch = self.epoch
            self._progress_phase = 'Complete'
            return
        proxy = None
        if self.config['vpn']['enabled']:
            from .vpn import ManagedVPN
            proxy = self._stack.enter_context(ManagedVPN(self.config['vpn']))
        self.source = DatasetSource(self.config['repo_url'], proxy=proxy)
        self.shards = self.source.list_shards()
        if not self.shards:
            raise RuntimeError('side-job dataset contains no training Parquet shards')

    def _start_epoch(self, cancelled=None):
        from .side_job_data import PrefetchedBatches, ShardDataset
        self._progress_epoch = self.epoch + 1
        self._progress_phase = 'Preparing'
        self.total_examples = self.total_batches = None
        self.epoch_batches = 0
        self.examples = self.correct = 0
        self.total_loss = 0.0
        name = self.next_shard if self.next_shard in self.shards else self.shards[self.epoch % len(self.shards)]
        self.shard_path = self.directory / 'train.parquet'
        self.shard_path.unlink(missing_ok=True)
        for partial in self.directory.glob('.shard-*.part'):
            partial.unlink(missing_ok=True)
        self.source.download_shard(name, self.shard_path, cancelled=cancelled)
        dataset = ShardDataset(self.shard_path, image_size=self.config['image_size'], seed=self.config['seed'] + self.epoch, passes=self.config.get('passes_per_shard', 1), cache_limit_mb=self.config.get('cache_limit_mb', 2048))
        self.total_examples = len(dataset)
        self.total_batches = (self.total_examples + self.config['batch_size'] - 1) // self.config['batch_size']
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.config['batch_size'], num_workers=0, pin_memory=True)
        prefetch = self.config.get('prefetch_batches', 2)
        self.iterator = PrefetchedBatches(loader, depth=prefetch) if prefetch else iter(loader)
        self.next_shard = self.shards[(self.shards.index(name) + 1) % len(self.shards)]
        self._progress_phase = 'Training'
        self.model.train()
        self._log('Side-job epoch %d (%d steps, %d examples, %d passes)', self.epoch + 1, self.total_batches, self.total_examples, self.config.get('passes_per_shard', 1))

    def _finish_epoch(self):
        if not self.examples:
            raise RuntimeError('side-job training shard is empty')
        self.epoch += 1
        save_checkpoint(self.directory, self._checkpoint())
        self._progress_phase = 'Complete'
        self._log('Side-job epoch %d complete: loss=%.4f accuracy=%.1f%% examples=%d', self.epoch, self.total_loss / self.examples, 100 * self.correct / self.examples, self.examples)
        self._clear_shard()

    def _clear_shard(self):
        self.pending_batch = None
        if self.iterator is not None:
            iterator, self.iterator = (self.iterator, None)
            if hasattr(iterator, 'close'):
                iterator.close()
            else:
                fetcher = getattr(iterator, '_dataset_fetcher', None)
                generator = getattr(fetcher, 'dataset_iter', None)
                if generator is not None and hasattr(generator, 'close'):
                    generator.close()
        if self.shard_path is not None:
            self.shard_path.unlink(missing_ok=True)
            self.shard_path = None

    def step(self, *, cancelled=None, pacer=None, stopped=None, deadline=None):
        if pacer is None:
            return self._step(cancelled=cancelled)
        with pacer.operations(stopped, deadline=deadline):
            return self._step(cancelled=cancelled)

    def _step(self, *, cancelled=None):
        if self.closed:
            raise RuntimeError('side-job is closed')
        if cancelled is not None and cancelled():
            return True
        if self.model is None:
            self._initialize()
        if self.config['epochs'] and self.epoch >= self.config['epochs']:
            return False
        if cancelled is not None and cancelled():
            return True
        if self.iterator is None:
            from .side_job_data import DatasetCancelled
            try:
                self._start_epoch(cancelled)
            except DatasetCancelled:
                self._clear_shard()
                return True
        if cancelled is not None and cancelled():
            return True
        try:
            if self.pending_batch is None:
                self.pending_batch = next(self.iterator)
        except StopIteration:
            self._finish_epoch()
            return not (self.config['epochs'] and self.epoch >= self.config['epochs'])
        if cancelled is not None and cancelled():
            return True
        images, labels = self.pending_batch
        self.pending_batch = None
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.config['use_amp']):
            logits = self.model(images)
            loss = nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError('side-job training produced a nonfinite loss')
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        count = labels.numel()
        self.epoch_batches += 1
        self.examples += count
        self.total_loss += loss.detach().item() * count
        self.correct += (logits.detach().argmax(dim=1) == labels).sum().item()
        return True

    def close(self):
        try:
            self._clear_shard()
        finally:
            self._stack.close()
            self.optimizer = self.scaler = self.model = self.source = None
            self.closed = True

def main(argv=None):
    from .config import load_config
    from .paths import default_config_path
    from .progress import LiveProgress
    from .telemetry import GpuUtilization
    parser = argparse.ArgumentParser(description='Train the side-job page-orientation CNN on CUDA')
    parser.add_argument('--config', type=Path, default=default_config_path())
    parser.add_argument('--epochs', type=int, help='total completed epochs, including resumed epochs; 0 repeats')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    job = None
    try:
        config = load_config(args.config)
        if args.epochs is not None:
            if args.epochs < 0:
                parser.error('--epochs must be nonnegative')
            config['side_job']['epochs'] = args.epochs
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        with ExitStack() as resources:
            telemetry_device = torch.device(config['device'])
            if telemetry_device.type == 'cuda' and telemetry_device.index is None:
                telemetry_device = torch.device('cuda', torch.cuda.current_device())
            gpu = resources.enter_context(GpuUtilization(str(telemetry_device)))
            progress = resources.enter_context(LiveProgress())
            job = SideJob(config, before_log=progress.clear)
            progress.update(job.status(gpu.read()), force=True)
            while job.step():
                progress.update(job.status(gpu.read()))
            progress.update(job.status(gpu.read()), force=True)
        return 0
    except KeyboardInterrupt:
        LOG.info('Stopped; the last completed epoch is saved')
        return 130
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        LOG.error('%s', exc)
        return 1
    finally:
        if job is not None:
            job.close()
if __name__ == '__main__':
    raise SystemExit(main())
