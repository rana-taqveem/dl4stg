"""Required test-set panels/GIFs; preserve baseline video IDs for the final model."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from q2_core import make_window, mirror, rotate, window_frame_numbers
from q2_experiment import save_json


def frame_array(x):
    return x.detach().cpu().numpy().squeeze()


def panel(rows, labels, title, path, columns=None):
    n = len(rows[0])
    fig, axes = plt.subplots(len(rows), n, figsize=(1.25 * n + 2, 1.35 * len(rows) + 0.5), squeeze=False)

    for r, row in enumerate(rows):
        for c, frame in enumerate(row):
            axes[r, c].imshow(frame_array(frame), cmap='gray', vmin=0, vmax=1)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(columns[c] if columns else str(c + 1), fontsize=9)
        axes[r, 0].set_ylabel(labels[r], fontsize=9)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0.02, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def gif_panel(rows, labels, title, path):
    """Keep all five context frames fixed; animate the five future frames."""
    font = ImageFont.truetype(font_manager.findfont('DejaVu Sans'), 12)
    frames = []

    for h in range(5):
        canvas = Image.new('RGB', (620, 50 + len(rows) * 76), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 5), f'{title} | horizon {h + 1}/5', fill='black', font=font)
        draw.text((160, 28), 'Five fixed input frames', fill='black', font=font)
        draw.text((510, 28), 'Future', fill='black', font=font)
        for r, row in enumerate(rows):
            top = 50 + r * 76
            draw.text((5, top + 22), labels[r], fill='black', font=font)
            for c in range(5):
                image = Image.fromarray((frame_array(row[c]) * 255).clip(0, 255).astype(np.uint8)).convert('RGB')
                canvas.paste(image, (160 + c * 64, top))
            image = Image.fromarray((frame_array(row[5 + h]) * 255).clip(0, 255).astype(np.uint8)).convert('RGB')
            canvas.paste(image, (520, top))
        frames.append(canvas)

    frames[0].save(path, save_all=True, append_images=frames[1:], duration=450, loop=0)


def component_boxes(frame):
    binary = (frame_array(frame) > .2).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return [s.tolist() for s in stats[1:] if s[cv2.CC_STAT_AREA] >= 12]


def separated(frame):
    boxes = component_boxes(frame)

    if len(boxes) != 2:
        return False

    a, b = boxes
    return (a[0] + a[2] < b[0] or b[0] + b[2] < a[0] or
            a[1] + a[3] < b[1] or b[1] + b[3] < a[1])


def choose_examples(experiment):
    path = experiment.out / 'example_selection.json'
    data = np.load(experiment.out / 'metrics' / 'A_seed0_stress.npz')
    worst = []

    for tag in ('M', 'R'):
        indices = np.argsort(-data['C' + tag], kind='stable')[:2]
        worst.extend({'transformation': tag, 'video': int(i + 9000),
                       'baseline_gap': float(data['C' + tag][i])} for i in indices)

    if path.exists():
        cached = json.loads(path.read_text())
        if cached['figure3'] != worst:
            raise RuntimeError('Baseline gaps changed after example selection. Remove outputs/example_selection.json and regenerate BOTH baseline and final figures.')
        if not all(9000 <= i < 10000 for i in cached['figure2_videos'] + [cached['speed_video']]):
            raise ValueError('Every displayed prediction example must be a test video.')
        return cached

    override_path = experiment.root / 'selection_review.json'
    overrides = json.loads(override_path.read_text()) if override_path.exists() else {}
    overlap = overrides.get('overlap_video')
    speed = overrides.get('speed_video')

    for index in (overlap, speed):
        if index is not None and (type(index) is not int or not 9000 <= index < 10000):
            raise ValueError('Example overrides must be global test IDs 9000-9999.')

    displayed = set()

    for k in (1, 2, 3):
        inputs, targets = window_frame_numbers(1, k)
        displayed.update(f - 1 for f in inputs + targets[:2])

    for index in range(9000, 10000):
        video = experiment.videos[index]
        counts = [len(component_boxes(frame)) for frame in video[:10]]
        if overlap is None and counts[0] == 2 and 1 in counts[1:9] and counts[-1] == 2:
            overlap = index
        if speed is None and all(separated(video[f]) for f in displayed):
            speed = index
        if overlap is not None and speed is not None:
            break

    if overlap is None or speed is None:
        raise RuntimeError('Could not select unambiguous examples. Set overlap_video and speed_video (global test IDs) in selection_review.json after inspecting the videos.')

    selected = [overlap] + [i for i in range(9000, 10000) if i != overlap][:2]
    result = {'figure2_videos': selected, 'overlap_video': overlap, 'speed_video': speed,
              'figure3': worst, 'selection_method': 'connected-component screening; visual confirmation required',
              'visual_confirmation': overrides.get('visual_confirmation', False)}
    result['speed_displayed_frame_numbers'] = sorted(f + 1 for f in displayed)
    save_json(path, result)
    return result


@torch.no_grad()
def generate_figures(experiment, final=False):
    selection = choose_examples(experiment)
    model = experiment.load_model('D' if final else 'A', 0, wrapped=final)

    def predict(x):
        return model(x.unsqueeze(0).to(experiment.device))[0].cpu()

    figures, gifs = experiment.out / 'figures', experiment.out / 'gifs'
    labels = ['X / f(X)', 'T(X) / T(Y)', 'T(X) / f(T(X))', 'T(X) / T(f(X))']
    caption_records = []

    for number, record in enumerate(selection['figure3'], 1):
        tag, index = record['transformation'], record['video']
        transform = mirror if tag == 'M' else rotate
        x, y = make_window(experiment.videos[index])
        base = predict(x)
        rows = [torch.cat((x, base)), torch.cat((transform(x), transform(y))),
                torch.cat((transform(x), predict(transform(x)))),
                torch.cat((transform(x), transform(base)))]
        stem = f'figure_{"6a" if final else "3"}_{number}_{tag}_video{index}'
        title = f'Figure {"6(a)" if final else "3"}, panel {number}: {"symmetrised D" if final else "A"}, {tag}, test video {index}'
        panel(rows, labels, title, figures / f'{stem}.png')
        gif_panel(rows, labels, title, gifs / f'{stem}.gif')
        caption_records.append({'figure': stem, 'video': index, 'transformation': tag,
                                'png': f'outputs/figures/{stem}.png', 'gif': f'outputs/gifs/{stem}.gif'})

    rows, row_labels = [], []
    speed_video = experiment.videos[selection['speed_video']]

    for k in (1, 2, 3):
        x, y = make_window(speed_video, k=k)
        rows.extend([torch.cat((x, predict(x)[:2])), torch.cat((x, y[:2]))])
        row_labels.extend([f'{k}x prediction', f'{k}x truth'])

    stem = 'figure_6b' if final else 'figure_5'
    panel(rows, row_labels, f'Figure {"6(b)" if final else "5"}: video {selection["speed_video"]}, '
          f'{"symmetrised D" if final else "A"}', figures / f'{stem}.png',
          ['Input 1', 'Input 2', 'Input 3', 'Input 4', 'Input 5', 'Horizon 1', 'Horizon 2'])

    if not final:
        rows, row_labels = [], []
        for index in selection['figure2_videos']:
            x, y = make_window(experiment.videos[index])
            rows.extend([torch.cat((x, predict(x))), torch.cat((x, y))])
            row_labels.extend([f'{index} prediction', f'{index} truth'])
        panel(rows, row_labels, f'Figure 2: model A, 1x; overlap-screened video {selection["overlap_video"]}', figures / 'figure_2.png')
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        data = np.load(experiment.out / 'metrics' / 'A_seed0_stress.npz')
        for ax, tag in zip(axes, ('M', 'R')):
            values = data['C' + tag]
            ax.hist(values, bins=30, color='steelblue', edgecolor='white')
            for item in selection['figure3']:
                if item['transformation'] == tag:
                    ax.axvline(item['baseline_gap'], linestyle='--', label=f'Video {item["video"]}')
            ax.set(xlabel=f'Per-video C_{tag}', ylabel='Number of test videos', title=f'{tag}: A seed 0, 1x')
            ax.legend()
        fig.suptitle('Figure 4. Consistency gaps over all 1,000 test videos')
        fig.savefig(figures / 'figure_4.png', dpi=160)
        plt.close(fig)

    save_json(experiment.out / f'figure_{"6" if final else "3"}_captions.json', caption_records)
    print('Saved final comparison figures.' if final else 'Saved Figures 2-5 and GIFs.', flush=True)
    return selection
