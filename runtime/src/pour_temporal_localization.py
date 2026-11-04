"""Bounded consensus of RGB detections; no scene coordinates enter decisions."""
from math import hypot, isfinite
from statistics import median


def consensus(observations, threshold, settings):
    rows = observations[-settings['window']:]
    usable = [(u, float(s)) for u, s in rows if isfinite(s) and s >= settings['score_floor']
              and len(u) == 2 and all(isfinite(float(v)) for v in u)]
    candidates = []
    for anchor, score in usable:
        if score < threshold:
            continue
        cluster = [(u, s) for u, s in usable if hypot(u[0]-anchor[0], u[1]-anchor[1]) <= settings['radius_px']]
        if len(cluster) < settings['votes'] or len(cluster) < settings['fraction'] * len(rows):
            continue
        uv = [median(u[k] for u, _ in cluster) for k in (0, 1)]
        candidates.append(dict(uv=uv, votes=len(cluster), anchor_score=score,
                               median_score=median(s for _, s in cluster)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x['votes'], x['anchor_score']), reverse=True)
    best = candidates[0]
    # Competing well-supported locations are ambiguous, even with high scores.
    if any(hypot(x['uv'][0]-best['uv'][0], x['uv'][1]-best['uv'][1]) > settings['radius_px']
           and x['votes'] >= best['votes'] for x in candidates[1:]):
        return None
    return best
