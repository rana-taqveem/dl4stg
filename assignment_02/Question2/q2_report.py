"""Tables 2-6, evidence-linked answers, and an optional single PDF report."""
import csv
import json
from pathlib import Path
import numpy as np
from q2_experiment import PROTOCOL, save_json


def fmt(value, scientific=False):
    value = float(value)
    return f'{value:.2e}' if scientific or (0 < abs(value) < 1e-3) else f'{value:.3g}'


def learned_entry(values, scientific=False):
    return f'{fmt(np.mean(values), scientific)} ({fmt(values[0], scientific)}, {fmt(values[1], scientific)})'


def read_metrics(out, name, seed):
    normal = out / 'metrics' / f'{name}_seed{seed}_normal.json'

    if not normal.exists():
        return None

    result = json.loads(normal.read_text())
    result.update({f'MSE{i + 1}': v for i, v in enumerate(result['MSE'])})
    stress = out / 'metrics' / f'{name}_seed{seed}_stress.json'

    if stress.exists():
        result.update(json.loads(stress.read_text()))

    return result


TABLE_COLUMNS = {
    2: ['MSE1', 'MSE2', 'MSE3', 'MSE4', 'MSE5', 'E5', 'parameters'],
    3: ['E5', 'CM', 'EM', 'CR', 'ER'],
    4: ['E2_1x', 'E2_2x', 'E2_3x', 'E5', 'E5_2x'],
    5: ['E5', 'CM', 'EM', 'CR', 'ER', 'E2_1x', 'E2_2x', 'E2_3x', 'parameters',
        'train_seconds_per_epoch', 'inference_seconds'],
}
TABLE_COLUMNS[6] = TABLE_COLUMNS[5]
TABLE_NAMES = {2: ['A', 'Blank', 'Copy-last'], 3: ['A', 'Blank', 'Copy-last'],
               4: ['A', 'Copy-last'], 5: ['A', 'B', 'C', 'D'], 6: ['sym_B', 'sym_D']}


def make_tables(out):
    tables = {}
    folder = out / 'tables'
    folder.mkdir(parents=True, exist_ok=True)

    for number, names in TABLE_NAMES.items():
        columns = TABLE_COLUMNS[number]
        rows = []
        complete = True
        for name in names:
            seeds = (0,) if name in ('Blank', 'Copy-last') else (0, 1)
            results = [read_metrics(out, name, s) for s in seeds]
            if any(r is None or any(c not in r for c in columns) for r in results):
                complete = False
                break
            row = [name]
            for column in columns:
                values = [r[column] for r in results]
                if column == 'parameters':
                    row.append(str(int(values[0])) if len(values) == 1 else f'{int(np.mean(values))} ({int(values[0])}, {int(values[1])})')
                    continue
                scientific = number == 6 and column in ('CM', 'CR')
                row.append(fmt(values[0], scientific) if len(values) == 1 else learned_entry(values, scientific))
            rows.append(row)
        if not complete:
            continue
        headers = ['Model'] + columns
        with (folder / f'table_{number}.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows)
        markdown = '| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n'
        markdown += '\n'.join('| ' + ' | '.join(row) + ' |' for row in rows)
        (folder / f'table_{number}.md').write_text(markdown, encoding='utf-8')
        tables[number] = (headers, rows, markdown)

    return tables


def mean_metric(out, name, key):
    return np.mean([read_metrics(out, name, s)[key] for s in (0, 1)])


def theory_answers(root):
    flow_path = root / 'outputs/setup/flow_summary.json'
    flow = json.loads(flow_path.read_text()) if flow_path.exists() else None
    q1 = ('Figure 1 must be generated from the notebook setup before completing this answer.' if flow is None else
          f'Figure 1 uses {flow["frame_pairs"]} consecutive frame pairs and {flow["kept_pixels"]:,} '
          f'retained pixel observations; median estimated speed is {fmt(flow["median_speed_pixels_per_frame"])} pixels/frame. '
          f'The 2x and 3x displacement references are {fmt(2 * flow["median_speed_pixels_per_frame"])} and '
          f'{fmt(3 * flow["median_speed_pixels_per_frame"])} pixels per observed interval. All direction bins are populated in '
          'the saved analysis, but unequal frequencies and the transformed-histogram diagnostics do not establish exact symmetry. '
          'Transformed directions are broadly represented; overlap errors, pixel weighting and the both-frames intensity mask '
          'could bias the distribution enough to affect conclusions about its degree of symmetry.')
    return {
        'Q2.1': q1,
        'Q2.2': 'Mirror reflects directions and strokes; rotation rotates directions and strokes; both preserve speed magnitudes and the square-box motion rule. '
                 'Speed-up changes observation intervals and motion displacements while preserving each retained frame\'s spatial appearance. '
                 'These interventions probe orientation/appearance sensitivity and temporal displacement, with later B/C/D comparisons separating explanations.',
        'Q2.3': 'One folded-grid cell represents 4 original pixels. Each baseline 3x3 gate convolution expands a path by one grid cell. '
                 'The longest within-step path starts at the new input or previous layer-1 hidden state, traverses the layer-1 gate convolution '
                 'and then the layer-2 gate convolution, giving 2 cells or 8 original pixels along either axis (Chebyshev radius). '
                 'The square support includes diagonal displacement (8,8), whose Euclidean distance is 8*sqrt(2)=11.3 pixels; we report both distance conventions explicitly. '
                 'The previous layer-2 state has a shorter one-convolution path. '
                 'The 1x1 output and pixel shuffle/unshuffle add no reach; recurrence does not add arbitrarily many spatial convolutions inside one step.',
        'Q2.6': 'Blank and Copy-last commute exactly with mirror and rotation, yet may predict the future poorly. '
                 'A small consistency gap therefore establishes agreement between transformed predictions, not correctness; transformed error must accompany it.',
        'Q2.10': 'Baseline speed tests alone confound unfamiliar displacement with architectural reach. '
                  'Compare A with speed-augmented C at the same architecture, then C with larger-reach D under identical speed augmentation and training protocol. '
                  'C improvement supports a data-exposure contribution; additional D improvement supports an architectural contribution, without proving reach is the only changed inductive bias.',
        'Q2.12': 'D keeps A\'s two 3x3 ConvLSTM layers but dilates the layer-2 gate convolution (dilation 3, padding 3); layer 1 is unchanged (dilation 1). '
                  'Using the Q2.3 conventions, the layer-1 convolution on [u, h1] moves information 1 folded cell and the layer-2 convolution on [h1, h2] moves it 3 cells. '
                  'The longest one-step path, from the new input or the previous layer-1 state through both gate convolutions, spans 1+3=4 cells, '
                  'or 16 original pixels per axis per step, strictly above A\'s 8; the path from the previous layer-2 state spans 3 cells (12 pixels). '
                  'Its maximum Euclidean distance is 16*sqrt(2)=22.6 pixels, versus A\'s 11.3. '
                  'Dilating only layer 2 avoids gridding: offsets {-1,0,1}+{-3,0,3} cover every cell offset from -4 to 4, whereas dilation 2 in both layers would only connect '
                  'cells an even number apart, splitting the 16x16 grid into four sub-grids that never exchange information. '
                  'D keeps the same 480,784 parameters (below 721,176), 9-step rollout, inputs, outputs and training protocol. No attention is used.',
    }


def generate_report(root, pdf=False):
    root = Path(root).resolve()
    out = root / 'outputs'
    tables = make_tables(out)
    reference_rows = []
    reference_columns = ['E5', 'E2_1x', 'CM', 'EM', 'CR', 'ER', 'E2_2x', 'E2_3x', 'E5_2x']

    for name in ('Blank', 'Copy-last'):
        values = read_metrics(out, name, 0)
        if values:
            reference_rows.append([name] + [fmt(values[c]) if c in values else 'pending' for c in reference_columns])

    if reference_rows:
        with (out / 'reference_results.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Model'] + reference_columns)
            writer.writerows(reference_rows)

    answers = theory_answers(root)
    pending = 'Pending the required trained-model measurements and visual inspection; no result is inferred from an untrained model.'

    for number in range(1, 18):
        answers.setdefault(f'Q2.{number}', pending)

    for key in ('Q2.5', 'Q2.11'):
        path = root / 'predictions' / f'{key.replace(".", "_")}.md'
        if path.exists():
            answers[key] = path.read_text(encoding='utf-8').split('\n\n', 1)[1].strip()

    if 2 in tables:
        a = read_metrics(out, 'A', 0)
        errors = np.mean([read_metrics(out, 'A', s)['MSE'] for s in (0, 1)], axis=0)
        answers['Q2.4'] = ('At original 1x, mean E(5) is ' + fmt(mean_metric(out, 'A', 'E5')) +
            ', compared with Blank ' + fmt(read_metrics(out, 'Blank', 0)['E5']) + ' and Copy-last ' + fmt(read_metrics(out, 'Copy-last', 0)['E5']) +
            '. Horizon errors are ' + ', '.join(fmt(v) for v in errors) + '. Figure 2 supplies the overlap example; its visual interpretation is recorded below.')

    if 3 in tables:
        em, er = mean_metric(out, 'A', 'EM'), mean_metric(out, 'A', 'ER')
        answers['Q2.8'] = (f'Model A at 1x has mirrored error {fmt(em)} and rotated error {fmt(er)}; '
            f'rotation is {"higher" if er > em else "not higher"} on this measure. '
            'Both transforms alter strokes and velocity direction. Figure 1 shows broad direction coverage but unequal frequencies, '
            'so this comparison alone cannot isolate appearance from motion-frequency effects; Table 5 provides the augmentation controls.')

    if 5 in tables:
        comparisons = []
        for name in ('B', 'C', 'D'):
            comparisons.append(f'{name}: original E(5)={fmt(mean_metric(out, name, "E5"))}, 3x E(2)={fmt(mean_metric(out, name, "E2_3x"))}')
        orientation = np.mean([mean_metric(out, 'B', k) - mean_metric(out, 'A', k) for k in ('EM', 'ER')])
        speed_delta = mean_metric(out, 'C', 'E2_3x') - mean_metric(out, 'A', 'E2_3x')
        reach_delta = mean_metric(out, 'D', 'E2_3x') - mean_metric(out, 'C', 'E2_3x')
        answers['Q2.13'] = ('Table 5 measurements: ' + '; '.join(comparisons) + '. '
            f'B-A mean mirror/rotation-error change is {orientation:+.3g}, '
            f'{"supporting an appearance/orientation-exposure contribution" if orientation < 0 else "not demonstrating an average appearance/orientation fix"}. '
            f'C-A 3x error change is {speed_delta:+.3g}, '
            f'{"supporting an unfamiliar-motion contribution" if speed_delta < 0 else "not demonstrating a 3x fix from exposure alone"}. '
            f'D-C 3x error change is {reach_delta:+.3g}, '
            f'{"supporting an architectural contribution" if reach_delta < 0 else "not demonstrating an advantage from this larger-reach design"}. '
            'Interpret A-to-B improvements in transformed error/gaps as evidence of orientation/appearance exposure; '
            'A-to-C improvements at faster speeds as evidence of unfamiliar motion; C-to-D improvements at 3x as evidence of architecture. '
            'If C remains poor while D improves, speed exposure alone was insufficient. More data does not enlarge the spatial dependency graph of one update. '
            'These are controlled comparisons, not proof that each failure has exactly one cause.')
        base = mean_metric(out, 'A', 'E5')
        answers['Q2.14'] = ('Original 1x E(5) changes relative to A: ' + ', '.join(
            f'{name}: {mean_metric(out, name, "E5") - base:+.3g}' for name in ('B', 'C', 'D')) +
            '. Positive changes indicate an accuracy cost. ' +
            ('D improves 3x accuracy, so any original-clip cost buys useful robustness for the assignment\'s faster-motion goal; acceptability depends on how often those conditions matter.'
             if reach_delta < 0 and mean_metric(out, 'D', 'E2_3x') < mean_metric(out, 'A', 'E2_3x') else
             'The D comparisons do not establish a 3x gain over both A and C, so an original-clip accuracy sacrifice cannot be justified by a demonstrated larger-reach benefit here.'))

    if 6 in tables and 5 in tables and 2 in tables and 3 in tables:
        answers['Q2.15'] = ('B learns a preference for symmetry through augmentation; it has no exact architectural guarantee. '
            'For the wrapper evaluated at TX, substitute t=gT for fixed T in G, so g=tT^-1 and g^-1=T t^-1. Then '
            'f_bar(TX)=1/8 sum_t T t^-1 f(tX)=T f_bar(X), because a pixel permutation is linear. '
            'Thus mirror and rotation gaps are exactly zero in real arithmetic, with floating-point residue in implementation. '
            f'Measured wrapped-D gaps are CM={fmt(mean_metric(out, "sym_D", "CM"), True)} and CR={fmt(mean_metric(out, "sym_D", "CR"), True)}. '
            f'B original E(5)={fmt(mean_metric(out, "B", "E5"))}; wrapped B={fmt(mean_metric(out, "sym_B", "E5"))}; '
            f'D={fmt(mean_metric(out, "D", "E5"))}; wrapped D={fmt(mean_metric(out, "sym_D", "E5"))}. '
            'Tables 5-6 report inference time including all eight passes, transforms, inverses and averaging. '
            f'Wrapping D changes original E(5) by {mean_metric(out, "sym_D", "E5") - mean_metric(out, "D", "E5"):+.3g}; '
            f'it {"hurts" if mean_metric(out, "sym_D", "E5") > mean_metric(out, "D", "E5") else "does not hurt"} original accuracy on this metric. '
            f'Mirror/rotation errors change by {mean_metric(out, "sym_D", "EM") - mean_metric(out, "D", "EM"):+.3g} and '
            f'{mean_metric(out, "sym_D", "ER") - mean_metric(out, "D", "ER"):+.3g}. '
            f'Measured inference multipliers are {mean_metric(out, "sym_B", "inference_seconds") / mean_metric(out, "B", "inference_seconds"):.3g} for B '
            f'and {mean_metric(out, "sym_D", "inference_seconds") / mean_metric(out, "D", "inference_seconds"):.3g} for D. '
            'Averaging misaligned or differently shaped plausible predictions can blur digits. '
            'The final model is symmetrised D, chosen as the required construction; optional D+B was not trained.')
        beats_references = all(mean_metric(out, 'sym_D', 'E5') < read_metrics(out, n, 0)['E5'] and
                             mean_metric(out, 'sym_D', 'E2_3x') < read_metrics(out, n, 0)['E2_3x'] for n in ('Blank', 'Copy-last'))
        answers['Q2.17'] = (('The errors support some learned predictive motion beyond the two references. ' if beats_references else
                            'The errors do not convincingly establish motion prediction beyond both references across normal and fast clips. ') +
            'Exact spatial equivariance alone does not establish learned motion. '
            f'Its 3x E(2) is {fmt(mean_metric(out, "sym_D", "E2_3x"))}, versus A at {fmt(mean_metric(out, "A", "E2_3x"))}; '
            'Tables 2-6 and matched examples must show useful forecasting beyond copying, and explain what speed exposure and larger reach contribute. '
            'Consistency and correctness should be assessed together across both seeds. The strongest remaining test would change digit appearance '
            'and motion jointly outside this benchmark, such as unseen shapes with acceleration or nonelastic collisions. '
            'Success on the specified transformations provides bounded evidence, not proof of a general physical motion rule.')

    notes_path = root / 'qualitative_notes.json'
    notes = json.loads(notes_path.read_text()) if notes_path.exists() else {}

    for question in ('Q2.4', 'Q2.7', 'Q2.9', 'Q2.13', 'Q2.14', 'Q2.15', 'Q2.16', 'Q2.17'):
        if notes.get(question):
            answers[question] = notes[question] if answers[question] == pending or question == 'Q2.17' else answers[question] + '\n\n' + notes[question]

    if len(answers['Q2.17'].split()) > 150:
        raise ValueError('Q2.17 exceeds the 150-word limit; shorten qualitative_notes.json.')

    selection_path = out / 'example_selection.json'
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    review_path = root / 'selection_review.json'
    review = json.loads(review_path.read_text()) if review_path.exists() else {}
    full_tables = set(tables) == {2, 3, 4, 5, 6}
    needs_notes = [q for q in ('Q2.4', 'Q2.7', 'Q2.9', 'Q2.16') if not notes.get(q)]
    figures_ready = (all((out / 'figures' / f'figure_{s}.png').exists() for s in ('2', '4', '5', '6b'))
                   and len(list((out / 'figures').glob('figure_3_*.png'))) == 4
                   and len(list((out / 'figures').glob('figure_6a_*.png'))) == 4
                   and len(list((out / 'gifs').glob('*.gif'))) >= 8)
    setup_ready = (out / 'setup/figure_1.png').exists() and (out / 'setup/table_1.csv').exists()
    ready = full_tables and not needs_notes and figures_ready and setup_ready and review.get('visual_confirmation', False)
    submission_path = root / 'submission.json'
    submission = json.loads(submission_path.read_text()) if submission_path.exists() else {}
    repository = submission.get('repository_url', 'Repository URL not configured in submission.json')
    status = {'all_tables_ready': full_tables, 'figures_ready': figures_ready, 'setup_ready': setup_ready,
            'missing_visual_answers': needs_notes, 'examples_visually_confirmed': review.get('visual_confirmation', False),
            'ready_for_pdf': ready}
    save_json(out / 'completion_status.json', status)
    lines = ['# Question 2 - Motion or Memory?',
           '**Status: ' + ('complete measurements and visual review' if ready else 'DRAFT: incomplete experiments or visual review') + '**',
           'Repository: ' + repository + ' (verify public visibility before submission).',
           'All learned entries are mean (seed 0, seed 1). Errors use three significant figures; small nonzero values use scientific notation. '
           'Training time is the mean train-pass wall time over 30 epochs, excluding downloads, validation passes and checkpoint I/O. '
           'Inference timings exclude host-to-device transfer and include the complete predictor; both timing conventions are logged.',
           'Final model: symmetrised D. D was specified before its experiments and not selected using test performance. All learned figures, GIFs and per-video histograms use seed 0.']
    protocol_path = out / 'protocol.json'

    if protocol_path.exists():
        lines.append('Runtime hardware: `' + json.dumps(json.loads(protocol_path.read_text())['hardware']) + '`')

    for name in ('A', 'B', 'C', 'D'):
        for seed in (0, 1):
            metrics = read_metrics(out, name, seed)
            if metrics and metrics.get('training_hardware'):
                lines.append(f'{name}, seed {seed}, training hardware (including resumed devices): `' + json.dumps(metrics['training_hardware']) + '`')

    setup_csv = out / 'setup/table_1.csv'

    if setup_csv.exists():
        with setup_csv.open(encoding='utf-8') as file:
            data = list(csv.reader(file))
        # One string: the report joins elements with blank lines, which would split the table.
        lines.extend(['## Table 1', '\n'.join(['| ' + ' | '.join(data[0]) + ' |', '|---|---|---|'] +
                                            ['| ' + ' | '.join(row) + ' |' for row in data[1:]])])

    for number in range(2, 7):
        lines += [f'## Table {number}', tables[number][2] if number in tables else 'Pending required final-epoch measurements for both seeds.']

    image_groups = {1: sorted((out / 'setup').glob('figure_1.png')),
                  2: sorted((out / 'figures').glob('figure_2.png')),
                  3: sorted((out / 'figures').glob('figure_3_*.png')),
                  4: sorted((out / 'figures').glob('figure_4.png')),
                  5: sorted((out / 'figures').glob('figure_5.png')),
                  6: sorted((out / 'figures').glob('figure_6*.png'))}

    for number, images in image_groups.items():
        lines += [f'## Figure {number}']
        lines += [f'![{image.stem}]({image.relative_to(out).as_posix()})' for image in images] or ['Pending trained predictions.']
        for image in images:
            gif = out / 'gifs' / f'{image.stem}.gif'
            if gif.exists():
                lines.append(f'Animated panel: [{gif.name}]({gif.relative_to(out).as_posix()}).')

    if selection:
        lines.append('Example selection (global dataset indices): `' + json.dumps(selection) + '`')

    for number in range(1, 18):
        lines += [f'## Q2.{number}', answers[f'Q2.{number}']]

    disclosure = root / 'AI_ASSISTANCE.md'
    lines += [disclosure.read_text(encoding='utf-8') if disclosure.exists() else
              '## AI assistance disclosure\n\nProvide the actual AI prompts, outputs and subsequent edits before submission.']
    report = out / 'question_2_report.md'
    report.write_text('\n\n'.join(lines) + '\n', encoding='utf-8')

    if pdf:
        if not ready:
            raise RuntimeError(f'PDF export requires completed experiments and reviewed examples: {status}. Draft Markdown is available at {report}.')
        export_pdf(out, answers, tables, image_groups)

    return status


def export_pdf(out, answers, tables, image_groups):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, A4
    from xml.sax.saxutils import escape
    from PIL import Image as PILImage

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(out / 'question_2_report.pdf'), pagesize=landscape(A4),
                          leftMargin=28, rightMargin=28, topMargin=26, bottomMargin=26)
    config_path = out.parent / 'submission.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    flow = [Paragraph('Question 2 - Motion or Memory?', styles['Title']),
          Paragraph(escape('Repository: ' + config.get('repository_url', 'Not configured')), styles['BodyText']),
          Paragraph('Final model: symmetrised D.', styles['BodyText'])]
    protocol = json.loads((out / 'protocol.json').read_text())
    flow.append(Paragraph(escape('Hardware: ' + json.dumps(protocol['hardware'])), styles['BodyText']))

    for name in ('A', 'B', 'C', 'D'):
        for seed in (0, 1):
            metrics = read_metrics(out, name, seed)
            if metrics and metrics.get('training_hardware'):
                flow.append(Paragraph(escape(f'{name}, seed {seed}, training hardware: ' + json.dumps(metrics['training_hardware'])), styles['BodyText']))

    for number in range(1, 18):
        flow += [Spacer(1, 10), Paragraph(f'Q2.{number}', styles['Heading2'])]
        for paragraph in answers[f'Q2.{number}'].split('\n\n'):
            flow.append(Paragraph(escape(paragraph), styles['BodyText']))

    for number in range(1, 7):
        flow += [PageBreak(), Paragraph(f'Table {number}', styles['Heading1'])]
        if number == 1:
            with (out / 'setup/table_1.csv').open(encoding='utf-8') as file:
                data = list(csv.reader(file))
        else:
            headers, rows, _ = tables[number]
            data = [headers] + rows
        # Paragraph wrapping keeps wide seed pairs inside their cells.
        from reportlab.lib.styles import ParagraphStyle
        small = ParagraphStyle('small', fontSize=6.5, leading=8)
        wrapped = [[Paragraph(escape(str(v)).replace('_', '_<br/>'), small) for v in row] for row in data]
        widths = [doc.width / len(data[0])] * len(data[0])
        table = Table(wrapped, colWidths=widths, repeatRows=1)
        table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eef4')),
                                   ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('GRID', (0, 0), (-1, -1), .3, colors.lightgrey)]))
        flow.append(table)

    captions = []

    for filename in ('figure_3_captions.json', 'figure_6_captions.json'):
        p = out / filename
        if p.exists():
            captions.extend(json.loads(p.read_text()))

    for number, images in image_groups.items():
        for image in images:
            flow += [PageBreak(), Paragraph(f'Figure {number}: {escape(image.stem)}', styles['Heading1'])]
            with PILImage.open(image) as im:
                width, height = im.size
            scale = min(doc.width / width, (doc.height - 100) / height)
            flow.append(Image(str(image), width=width * scale, height=height * scale))
            if number == 1:
                caption = (out / 'setup/figure_1_caption.md').read_text().replace('**', '')
                flow.append(Paragraph(escape(caption), styles['BodyText']))
            for caption in captions:
                if caption['figure'] == image.stem:
                    flow.append(Paragraph(escape(f'Video {caption["video"]}; {caption["transformation"]}; GIF: {caption["gif"]}'), styles['BodyText']))

    flow += [PageBreak(), Paragraph('AI assistance disclosure', styles['Heading1'])]
    disclosure = out.parent / 'AI_ASSISTANCE.md'

    if disclosure.exists():
        for paragraph in disclosure.read_text(encoding='utf-8').split('\n\n'):
            flow.append(Paragraph(escape(paragraph), styles['BodyText']))

    doc.build(flow)
