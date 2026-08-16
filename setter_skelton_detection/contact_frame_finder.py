"""
Semi-automated helper for locating the frame where the setter touches the ball,
to speed up the manual scrubbing step in `making dataset.ipynb`.

It does NOT pick a single "correct" frame automatically — ball tracking on a
handheld, variable-lighting video isn't reliable enough for that (verified: on
a test clip, the estimated contact frame landed within ~5 frames of the frame
a human picked, but that accuracy isn't guaranteed across different videos).
Instead, `show_contact_sheet()` displays a small grid of candidate frames
around a rough time window, with the ball-tracking heuristic's best guess(es)
highlighted, so a human only has to glance at ~12 thumbnails and pick one
instead of stepping through `frame()` one frame at a time.

Usage (same rough-time-then-refine workflow as the existing notebook):
    from contact_frame_finder import show_contact_sheet
    show_contact_sheet('VNL2025/videos/IMG_5243.MOV', start_sec=109, end_sec=113)
    # -> look at the grid, note the frame number that shows ball-hand contact
    # -> pass that number as target_frame to the existing frame() function

If the ball isn't being picked up (no red-highlighted candidates), the HSV
color range likely needs adjusting for this video's lighting/ball color via
the hsv_lower/hsv_upper arguments.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def _detect_ball_in_frame(frame, prev_gray, top_ignore_frac, hsv_lower, hsv_upper,
                           min_area, max_area, min_circularity):
    """Find the best ball-colored, moving, roughly-circular blob in one frame."""
    h, _ = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    mask[: int(h * top_ignore_frac), :] = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        # ball color alone is not distinctive enough (ads/jerseys/crowd match it);
        # requiring recent motion filters out static yellow objects in the background
        diff = cv2.absdiff(gray, prev_gray)
        _, motion_mask = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.dilate(motion_mask, np.ones((7, 7), np.uint8))
        mask = cv2.bitwise_and(mask, motion_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        circularity = area / (np.pi * r * r + 1e-6)
        if circularity < min_circularity:
            continue
        if best is None or area > best[2]:
            best = (x, y, area)
    return gray, best


def _build_tracklets(detections, max_speed=25, max_gap=2):
    """Greedy nearest-neighbor linking of per-frame ball detections into short
    tracklets of smooth motion, so a single stray yellow blob (crowd, ad board)
    doesn't get treated the same as a genuine multi-frame ball trajectory."""
    frames = sorted(detections.keys())
    tracklets = []
    active = []  # (last_frame, last_x, last_y, tracklet_index)

    for fn in frames:
        used = set()
        new_active = []
        for (last_fn, lx, ly, idx) in active:
            gap = fn - last_fn
            if gap > max_gap:
                continue
            best_i, best_dist = None, None
            for i, (x, y, _) in enumerate(detections[fn]):
                if i in used:
                    continue
                dist = np.hypot(x - lx, y - ly)
                if dist <= max_speed * gap and (best_dist is None or dist < best_dist):
                    best_i, best_dist = i, dist
            if best_i is not None:
                used.add(best_i)
                x, y, _ = detections[fn][best_i]
                tracklets[idx].append((fn, x, y))
                new_active.append((fn, x, y, idx))
            else:
                new_active.append((last_fn, lx, ly, idx))
        active = new_active

        for i, (x, y, _) in enumerate(detections[fn]):
            if i in used:
                continue
            tracklets.append([(fn, x, y)])
            active.append((fn, x, y, len(tracklets) - 1))

    return tracklets


def suggest_contact_frames(video_path, start_sec, end_sec, fps=None, crop_bottom=600,
                            hsv_lower=(10, 60, 140), hsv_upper=(35, 255, 255),
                            top_ignore_frac=0.42, min_area=10, max_area=400,
                            min_circularity=0.45, max_speed=25, max_gap=2,
                            min_tracklet_len=6, n_suggestions=3):
    """Return up to n_suggestions candidate frame numbers, ranked by tracklet
    length (longer = more likely a genuine ball flight, not noise). Within each
    tracklet, the suggested frame is where the ball is lowest on screen (max y),
    i.e. the point right before it gets redirected upward by the set.
    """
    cap = cv2.VideoCapture(video_path)
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame, end_frame = int(start_sec * fps), int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    detections = {}
    prev_gray = None
    for fn in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[:-crop_bottom, :, :] if crop_bottom else frame
        prev_gray, best = _detect_ball_in_frame(
            frame, prev_gray, top_ignore_frac, hsv_lower, hsv_upper,
            min_area, max_area, min_circularity)
        if best is not None:
            detections.setdefault(fn, []).append(best)
    cap.release()

    tracklets = _build_tracklets(detections, max_speed=max_speed, max_gap=max_gap)
    tracklets = [t for t in tracklets if len(t) >= min_tracklet_len]
    tracklets.sort(key=len, reverse=True)

    suggestions = []
    for t in tracklets[:n_suggestions]:
        ys = [p[2] for p in t]
        peak_idx = int(np.argmax(ys))
        suggestions.append(int(t[peak_idx][0]))
    return suggestions


def show_contact_sheet(video_path, start_sec, end_sec, fps=None, crop_bottom=600,
                        n_thumbs=12, hsv_lower=(10, 60, 140), hsv_upper=(35, 255, 255),
                        n_suggestions=3, figscale=3.2):
    """Display an evenly-spaced grid of frames across [start_sec, end_sec], with
    ball-tracking's suggested contact frame(s) outlined in red, so a human can
    pick the true contact frame from one glance instead of stepping through
    frame() one frame at a time.
    """
    cap = cv2.VideoCapture(video_path)
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame, end_frame = int(start_sec * fps), int(end_sec * fps)
    cap.release()

    suggestions = suggest_contact_frames(
        video_path, start_sec, end_sec, fps=fps, crop_bottom=crop_bottom,
        hsv_lower=hsv_lower, hsv_upper=hsv_upper, n_suggestions=n_suggestions)

    thumb_frames = sorted(set(
        np.linspace(start_frame, end_frame - 1, n_thumbs, dtype=int).tolist() + suggestions
    ))

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    images = {}
    for fn in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        if fn in thumb_frames:
            frame = frame[:-crop_bottom, :, :] if crop_bottom else frame
            images[fn] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()

    n = len(thumb_frames)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    _fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * figscale, n_rows * figscale))
    axes = np.atleast_1d(axes).flatten()

    for ax, fn in zip(axes, thumb_frames):
        if fn not in images:
            ax.axis('off')
            continue
        ax.imshow(images[fn])
        is_suggested = fn in suggestions
        title = f'frame {fn}' + (' *suggested*' if is_suggested else '')
        ax.set_title(title, color='red' if is_suggested else 'black',
                      fontweight='bold' if is_suggested else 'normal')
        if is_suggested:
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(4)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    return thumb_frames, suggestions


if __name__ == '__main__':
    # quick manual check against a clip with a known contact frame (6672)
    video_path = 'VNL2025/videos/IMG_5243.MOV'
    fps = 59.96765943782469
    thumbs, suggestions = show_contact_sheet(video_path, 6580 / fps, 6720 / fps, fps=fps)
    print('thumbnails shown:', thumbs)
    print('suggested frames:', suggestions, '(true contact frame was 6672)')
