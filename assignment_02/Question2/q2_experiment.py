"""Resumable assignment protocol. Real results always require all 30 epochs."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
import numpy as np
import torch
from q2_models import build_model, count_parameters, Symmetrised
from q2_data import seed_everything, loader, per_horizon_mse, per_video_mse
from q2_core import mirror, rotate

PROTOCOL = {'epochs': 30, 'batch_size': 32, 'learning_rate': 1e-3,
            'seeds': [0, 1], 'training_videos': 8000, 'test_videos': 1000,
            'rollout_steps': 9, 'version': 1}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def hardware(device):
    return {'platform': platform.platform(), 'processor': platform.processor(),
            'torch': torch.__version__, 'device': str(device),
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'cpu_threads': torch.get_num_threads()}


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True,
                                   stderr=subprocess.STDOUT).strip()


def require_committed_prediction(root, question, text):
    """Write once and commit ONLY this prediction, preserving other staged work."""
    root = Path(root).resolve()
    path = root / 'predictions' / f'{question}.md'
    record_path = root / 'outputs' / 'prediction_records' / f'{question}.json'
    previous = json.loads(record_path.read_text()) if record_path.exists() else None

    if previous and path.exists():
        content_hash = hashlib.sha256(path.read_text(encoding='utf-8').strip().encode()).hexdigest()
        valid = content_hash == previous['content_sha256'] if 'content_sha256' in previous else hashlib.sha256(path.read_bytes()).hexdigest() == previous['sha256']
        if not valid:
            raise RuntimeError(f'{question} changed after its prediction record was frozen; preserve the original prediction.')

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'# {question}: pre-experiment prediction\n\n{text}\n', encoding='utf-8')

    try:
        repo = Path(git(root, 'rev-parse', '--show-toplevel')).resolve()
        relative = path.relative_to(repo).as_posix()
        content = path.read_text(encoding='utf-8').strip()
        try:
            commit = git(repo, 'log', '-1', '--format=%H', '--', relative)
            committed = git(repo, 'show', f'{commit}:{relative}') if commit else ''
        except subprocess.CalledProcessError:
            committed = ''
        if committed.strip() != content:
            git(repo, 'add', '--', relative)
            git(repo, 'commit', '--only', '-m', f'Record {question} before corresponding experiments', '--', relative)
            commit = git(repo, 'log', '-1', '--format=%H', '--', relative)
        assert git(repo, 'show', f'{commit}:{relative}').strip() == content
    except (subprocess.CalledProcessError, ValueError, AssertionError) as exc:
        raise RuntimeError(f'{question} must be committed before continuing. Commit {path} in your repository, then rerun. Git detail: {exc}') from exc

    if previous and previous['commit'] != commit:
        raise RuntimeError(f'{question} prediction commit changed after its record was frozen.')

    record = previous or {'question': question, 'commit': commit,
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'verified_before_stage_utc': datetime.now(timezone.utc).isoformat()}
    record['content_sha256'] = hashlib.sha256(path.read_text(encoding='utf-8').strip().encode()).hexdigest()
    save_json(record_path, record)
    return record


Q25 = ('I predict that model A will not be exactly consistent under either mirror or rotation, '
       'with rotation likely causing a larger accuracy loss because it changes digit appearance more strongly. '
       'Figure 1 shows motion in every direction bin, although unequal frequencies and flow bias prevent assuming exact motion symmetry. '
       'I expect 2x to degrade prediction and 3x to be particularly difficult: the median displacement references are 6.64 and 9.95 pixels per observed interval, '
       'while the longest one-step path in A spans 8 original pixels; reach is a bound, not a guarantee of reliable tracking.')


class Experiment:
    def __init__(self, root, videos, device=None):
        self.root = Path(root).resolve()
        self.out = self.root / 'outputs'
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.videos = videos

        if len(videos) != 10_000:
            raise ValueError('Full protocol needs 10,000 videos.')

        for folder in ('checkpoints', 'metrics', 'figures', 'tables', 'gifs', 'prediction_records'):
            (self.out / folder).mkdir(parents=True, exist_ok=True)

        save_json(self.out / 'protocol.json', {'protocol': PROTOCOL, 'hardware': hardware(self.device),
                  'final_model': 'symmetrised D', 'D_design': 'layer-1 gates dilation=1, padding=1; layer-2 gates dilation=3, padding=3; axial/Chebyshev reach=16, Euclidean max=22.6274; full grid connectivity; parameters=480784'})

    def checkpoint_path(self, name, seed):
        return self.out / 'checkpoints' / f'{name}_seed{seed}.pt'

    def read_checkpoint(self, name, seed):
        checkpoint = torch.load(self.checkpoint_path(name, seed), map_location=self.device, weights_only=False)

        if checkpoint['protocol'] != PROTOCOL or checkpoint['model_name'] != name or checkpoint['seed'] != seed:
            raise RuntimeError('Checkpoint protocol/model/seed does not match this experiment.')

        return checkpoint

    def load_model(self, name, seed=0, wrapped=False):
        seed_everything(seed)
        model = build_model(name).to(self.device)

        if name not in ('Blank', 'Copy-last'):
            checkpoint = self.read_checkpoint(name, seed)
            if checkpoint['epoch'] != 30:
                raise RuntimeError(f'{name} seed {seed} is not trained for all 30 epochs.')
            model.load_state_dict(checkpoint['model'])

        if wrapped:
            model = Symmetrised(model)

        return model.eval()

    def train(self, name, seed):
        if name not in ('A', 'B', 'C', 'D') or seed not in (0, 1):
            raise ValueError('Required models are A-D, with exactly seeds 0 and 1.')

        if name != 'A':
            self.require_q211()

        seed_everything(seed)
        model = build_model(name).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        start, history = 0, []

        if self.checkpoint_path(name, seed).exists():
            ckpt = self.read_checkpoint(name, seed)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start, history = ckpt['epoch'], ckpt['history']

        for epoch in range(start, 30):
            model.train()
            sync(self.device)
            began = time.perf_counter()
            total, seen = 0., 0
            for x, y, mask in loader(self.videos, self.device, 'train', name, seed, epoch):
                x, y, mask = (v.to(self.device, non_blocking=True) for v in (x, y, mask))
                optimizer.zero_grad(set_to_none=True)
                losses = per_video_mse(model(x), y, mask)
                loss = losses.mean()
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f'Nonfinite loss in {name} seed {seed}, epoch {epoch + 1}.')
                loss.backward()
                optimizer.step()
                total += losses.detach().sum().item()
                seen += len(x)
            sync(self.device)
            elapsed = time.perf_counter() - began
            assert seen == 8000
            # Monitoring only; final weights are always retained, never best-val weights.
            model.eval()
            val_total = 0.
            with torch.no_grad():
                for x, y, mask in loader(self.videos, self.device, 'validation'):
                    x, y, mask = (v.to(self.device) for v in (x, y, mask))
                    val_total += per_video_mse(model(x), y, mask).sum().item()
            history.append({'epoch': epoch + 1, 'train_mse': total / 8000,
                            'validation_mse': val_total / 1000, 'training_seconds': elapsed,
                            'hardware': hardware(self.device)})
            ckpt = {'protocol': PROTOCOL, 'model_name': name, 'seed': seed, 'epoch': epoch + 1,
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'history': history, 'hardware': hardware(self.device)}
            temp = self.checkpoint_path(name, seed).with_suffix('.tmp')
            torch.save(ckpt, temp)
            temp.replace(self.checkpoint_path(name, seed))
            save_json(self.out / 'metrics' / f'{name}_seed{seed}_training.json', history)
            print(f'{name} seed={seed} epoch={epoch + 1}/30 train={total / 8000:.6g} val={val_total / 1000:.6g} seconds={elapsed:.1f}', flush=True)

        return model.eval()

    @torch.no_grad()
    def evaluate_normal(self, name, seed=0, wrapped=False):
        label = f'sym_{name}' if wrapped else name
        model = self.load_model(name, seed, wrapped)
        errors, seconds = [], 0.

        for x, y, mask in loader(self.videos, self.device):
            x, y = x.to(self.device), y.to(self.device)
            sync(self.device)  # Timing starts only after the batch is on-device.
            began = time.perf_counter()
            prediction = model(x)
            sync(self.device)
            seconds += time.perf_counter() - began
            errors.append(per_horizon_mse(prediction, y).cpu().numpy())

        errors = np.concatenate(errors)
        assert errors.shape == (1000, 5)
        np.savez_compressed(self.out / 'metrics' / f'{label}_seed{seed}_normal.npz', mse=errors,
                            dataset_indices=np.arange(9000, 10000))
        training = self.read_checkpoint(name, seed) if name not in ('Blank', 'Copy-last') else None
        summary = {'model': label, 'seed': seed, 'MSE': errors.mean(0).tolist(),
                   'E5': float(errors.mean()), 'E2_1x': float(errors[:, :2].mean()),
                   'parameters': count_parameters(model), 'inference_seconds': seconds,
                   'train_seconds_per_epoch': float(np.mean([r['training_seconds'] for r in training['history']])) if training else 0,
                   'hardware': hardware(self.device),
                   'training_hardware': list({json.dumps(r.get('hardware', training['hardware']), sort_keys=True):
                                               r.get('hardware', training['hardware']) for r in training['history']}.values()) if training else None}
        save_json(self.out / 'metrics' / f'{label}_seed{seed}_normal.json', summary)
        print(f'{label} seed {seed}: E5={summary["E5"]:.6g}', flush=True)
        return summary

    def require_q25(self):
        return require_committed_prediction(self.root, 'Q2_5', Q25)

    def require_q211(self):
        # This record is formed only AFTER the baseline stress results exist.
        errors = [json.loads((self.out / 'metrics' / f'A_seed{s}_stress.json').read_text()) for s in (0, 1)]
        cm = np.mean([v['CM'] for v in errors])
        cr = np.mean([v['CR'] for v in errors])
        e3 = np.mean([v['E2_3x'] for v in errors])
        text = (f'After observing baseline A mean gaps CM={cm:.3g}, CR={cr:.3g}, and 3x E(2)={e3:.3g}, '
                'I predict symmetry-augmented B will reduce orientation-related inconsistency and transformed error, without guaranteeing exact equivariance. '
                'I predict speed-augmented C will improve faster-motion prediction but may remain limited at 3x by its 8-pixel per-axis reach. '
                'I predict D, with the same speed augmentation and 16-pixel per-axis reach, will improve 3x tracking relative to C, while not guaranteeing mirror or rotation consistency.')
        return require_committed_prediction(self.root, 'Q2_11', text)

    @torch.no_grad()
    def evaluate_stress(self, name, seed=0, wrapped=False):
        self.require_q25()
        label = f'sym_{name}' if wrapped else name
        model = self.load_model(name, seed, wrapped)
        records = {k: [] for k in ('CM', 'EM', 'CR', 'ER')}

        for x, y, _ in loader(self.videos, self.device):
            x, y = x.to(self.device), y.to(self.device)
            base = model(x)
            for tag, transform in (('M', mirror), ('R', rotate)):
                transformed = model(transform(x))
                records['C' + tag].append(per_horizon_mse(transformed, transform(base)).mean(1).cpu().numpy())
                records['E' + tag].append(per_horizon_mse(transformed, transform(y)).mean(1).cpu().numpy())

        arrays = {k: np.concatenate(v) for k, v in records.items()}

        for speed in (2, 3):
            values = []
            for x, y, mask in loader(self.videos, self.device, speed=speed):
                x, y = x.to(self.device), y.to(self.device)
                error = per_horizon_mse(model(x), y).cpu().numpy()
                error[~mask.numpy()] = np.nan
                values.append(error)
            arrays[f'mse_{speed}x'] = np.concatenate(values)

        assert all(len(v) == 1000 for v in arrays.values())
        np.savez_compressed(self.out / 'metrics' / f'{label}_seed{seed}_stress.npz', **arrays)
        result = {k: float(arrays[k].mean()) for k in records}
        result.update(E2_2x=float(arrays['mse_2x'][:, :2].mean()),
                      E2_3x=float(arrays['mse_3x'][:, :2].mean()),
                      E5_2x=float(arrays['mse_2x'].mean()))
        save_json(self.out / 'metrics' / f'{label}_seed{seed}_stress.json', result)
        return result

    def baseline(self):
        for name in ('Blank', 'Copy-last'):
            self.evaluate_normal(name)

        for seed in (0, 1):
            self.train('A', seed)
            self.evaluate_normal('A', seed)

    def stress(self):
        self.require_q25()

        for name in ('Blank', 'Copy-last', 'A'):
            for seed in ((0, 1) if name == 'A' else (0,)):
                self.evaluate_stress(name, seed)

    def improvements(self):
        self.require_q211()

        for name in ('B', 'C', 'D'):
            for seed in (0, 1):
                self.train(name, seed)
                self.evaluate_normal(name, seed)
                self.evaluate_stress(name, seed)

    def final(self):
        for name in ('B', 'D'):
            for seed in (0, 1):
                self.evaluate_normal(name, seed, wrapped=True)
                self.evaluate_stress(name, seed, wrapped=True)

