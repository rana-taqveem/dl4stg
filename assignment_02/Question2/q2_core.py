"""Shared transformations; Table 1 and experiments import the same functions."""
import torch


def mirror(x):
    """M(x)[..., i, j] = x[..., i, 63-j] for a 64x64 frame."""
    return torch.flip(x, dims=(-1,))


def rotate(x, a=1):
    """R^a: a anticlockwise quarter-turns on the spatial axes."""
    return torch.rot90(x, k=a, dims=(-2, -1))


G = tuple((a, mirrored) for mirrored in (False, True) for a in range(4))


def apply_symmetry(x, g):
    """Apply R^a or MR^a, rotating before mirroring."""

    if g not in G:
        raise ValueError("g must be one of the eight elements in G.")

    a, mirrored = g
    rotated = rotate(x, a)
    return mirror(rotated) if mirrored else rotated


def inverse_symmetry(x, g):
    """Undo g: rotations invert their angle; each MR^a is self-inverse."""

    if g not in G:
        raise ValueError("g must be one of the eight elements in G.")

    a, mirrored = g
    return apply_symmetry(x, g) if mirrored else rotate(x, -a)


def allowed_starts(k=1, speed_augmented=False):
    """Allowed one-based training starts, inclusive via a Python range."""

    if type(k) is not int or k not in (1, 2, 3):
        raise ValueError("k must be an integer in {1, 2, 3}.")

    if not speed_augmented:
        if k != 1:
            raise ValueError("Normal-speed A/B training requires k=1.")
        return range(1, 12)  # t=1,...,11; all five targets exist.

    return range(1, 21 - 6 * k)  # t+6k <= 20: at least two targets.


def window_frame_numbers(t=1, k=1):
    """Return one-based frame numbers for five inputs and available targets."""

    if type(k) is not int or k not in (1, 2, 3):
        raise ValueError("k must be an integer in {1, 2, 3}.")

    if type(t) is not int or not 1 <= t <= 20 - 6 * k:
        raise ValueError("Window requires five inputs and at least two targets within frames 1-20.")

    inputs = tuple(t + j * k for j in range(5))
    targets = tuple(t + j * k for j in range(5, 10) if t + j * k <= 20)
    return inputs, targets


def make_window(video, t=1, k=1):
    """Return X: (5,1,64,64), Y: (H,1,64,64), where 2 <= H <= 5."""

    if tuple(video.shape) != (20, 1, 64, 64):
        raise ValueError("Expected one complete video with shape (20,1,64,64).")

    inputs, targets = window_frame_numbers(t, k)
    input_indices = torch.tensor(inputs, device=video.device) - 1
    target_indices = torch.tensor(targets, device=video.device) - 1
    return video.index_select(0, input_indices), video.index_select(0, target_indices)


def sample_training_window(video, generator, speed_augmented=False,
                           symmetry_augmented=False):
    """Sample speed, then start, then symmetry; use a seeded CPU Generator.

    A: both flags False; B: symmetry only; C/D: speed only.
    Optional D+B: both flags True. Apply the same symmetry to X and Y.
    """
    k = int(torch.randint(1, 4, (), generator=generator).item()) if speed_augmented else 1
    starts = allowed_starts(k, speed_augmented)
    t = int(torch.randint(starts.start, starts.stop, (), generator=generator).item())
    g = G[int(torch.randint(0, 8, (), generator=generator).item())] if symmetry_augmented else G[0]
    x, y = make_window(video, t=t, k=k)
    return apply_symmetry(x, g), apply_symmetry(y, g), {"t": t, "k": k, "g": g}

