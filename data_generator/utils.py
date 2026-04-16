import math
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kirigami_x_mplconfig")

import numpy as np
from matplotlib.path import Path


RIDGE_THRESHOLD = 1e12
RIDGE_LAMBDA = 1e-8
MAX_COND = 1e20
MASK_MIN_FILL = 0.03
MASK_MAX_FILL = 0.97
MAX_OVERLAP = 0.02


def is_horizontal(i, j):
    return (i % 2) != (j % 2)


def node_id(rows, cols, i, j, k):
    if 0 <= i < rows and 0 <= j < cols:
        if j > 0 and k == 0:
            j -= 1
            k = 2
        if i > 0 and k == 3:
            i -= 1
            k = 1
        step = 3 - int(i > 0)
        base = i * (2 * cols + 1) + (cols if i > 0 else 0)
        shift = j * step + (1 if j > 0 else 0)
        return base + shift + k - (1 if j > 0 else 0)

    bulk_count = node_id(rows, cols, rows - 1, cols - 1, 2) + 1
    per_side = [rows + 1, cols + 1, rows + 1, cols + 1]

    if j == -1:
        side, b, inner, valid = 0, i, (i, 0), [3, 2, 1]
    elif i == rows:
        side, b, inner, valid = 1, j, (rows - 1, j), [0, 3, 2]
    elif j == cols:
        side, b, inner, valid = 2, rows - 1 - i, (i, cols - 1), [1, 0, 3]
    elif i == -1:
        side, b, inner, valid = 3, cols - 1 - j, (0, j), [2, 1, 0]
    else:
        return None

    if k not in valid:
        return None
    if k == valid[1]:
        return node_id(rows, cols, inner[0], inner[1], (k + 2) % 4)
    if k == valid[0]:
        b -= 1
    return bulk_count + sum(per_side[:side]) + b + 1


def outer_boundary_linkages(rows, cols, side):
    if side == 0:
        return [(i, -1) for i in range(rows)]
    if side == 1:
        return [(rows, j) for j in range(cols)]
    if side == 2:
        return [(i, cols) for i in range(rows - 1, -1, -1)]
    return [(-1, j) for j in range(cols - 1, -1, -1)]


def inner_boundary_linkages(rows, cols, side):
    if side == 0:
        return [(i, 0) for i in range(rows)]
    if side == 1:
        return [(rows - 1, j) for j in range(cols)]
    if side == 2:
        return [(i, cols - 1) for i in range(rows - 1, -1, -1)]
    return [(0, j) for j in range(cols - 1, -1, -1)]


def parallel_to_boundary(i, j, side):
    horizontal = is_horizontal(i, j)
    return (horizontal and side % 2 == 1) or ((not horizontal) and side % 2 == 0)


def outer_boundary_node_ids(rows, cols, side):
    out = []
    for i, j in outer_boundary_linkages(rows, cols, side):
        if parallel_to_boundary(i, j, side):
            out.append(node_id(rows, cols, i, j, (side + 2) % 4))
    return out


def build_linkages(rows, cols):
    return np.array(
        [[node_id(rows, cols, i, j, k) for k in range(4)] for i in range(rows) for j in range(cols)],
        dtype=int,
    )


def build_quads(rows, cols):
    quads = []
    for i in range(rows + 1):
        for j in range(cols):
            cur = [node_id(rows, cols, i, j, k) for k in range(4)]
            top = [node_id(rows, cols, i - 1, j, k) for k in range(4)]
            right = [node_id(rows, cols, i, j + 1, k) for k in range(4)]
            left = [node_id(rows, cols, i, j - 1, k) for k in range(4)]
            q0 = [left[3], cur[0], cur[3], top[0]]
            q1 = [cur[3], cur[2], right[3], top[2]]
            if is_horizontal(i, j):
                q1 = np.roll(q1, 1).tolist()
            else:
                q0 = np.roll(q0, 1).tolist()
            quads.append(q0)
            if j == cols - 1:
                quads.append(q1)
    return np.array(quads, dtype=int)


def build_linkage_to_quads(linkages, quads):
    node_to_quads = [[] for _ in range(int(quads.max()) + 1)]
    for qi, quad in enumerate(quads):
        for node in quad:
            node_to_quads[node].append(qi)

    out = [[] for _ in range(len(linkages))]
    for li, linkage in enumerate(linkages):
        for pos in range(4):
            node = linkage[pos]
            prev = linkage[(pos - 1) % 4]
            for qi in node_to_quads[node]:
                quad = quads[qi]
                if node in quad and prev in quad:
                    out[li].append(qi)
    return out


def design_matrix(rows, cols, x_matrix):
    # Each x_ij = a / b places one linkage corner x base-lengths along its seed segment.
    # The opposite free corner sits one more base-length farther on the same line.
    if x_matrix.shape != (rows, cols):
        raise ValueError(f"x_matrix must have shape {(rows, cols)}")

    seed_count = rows + cols + 4
    point_count = rows + cols + 2 * rows * cols + 2 * (rows + cols + 2)
    mat = np.zeros((point_count, seed_count), dtype=np.float64)

    for j in range(cols):
        mat[node_id(rows, cols, 0, j, 3), j] = 1.0
    for i in range(rows):
        mat[node_id(rows, cols, i, 0, 0), cols + i] = 1.0

    shift = rows + cols
    mat[node_id(rows, cols, 0, -1, 3), shift + 0] = 1.0
    mat[node_id(rows, cols, rows, 0, 0), shift + 1] = 1.0
    mat[node_id(rows, cols, rows - 1, cols, 1), shift + 2] = 1.0
    mat[node_id(rows, cols, -1, cols - 1, 2), shift + 3] = 1.0

    for i in range(rows):
        for j in range(cols):
            x = float(x_matrix[i, j])
            if is_horizontal(i, j):
                a, b = 0, 3
                c1, c2 = x, x + 1.0
            else:
                a, b = 3, 0
                c1, c2 = x + 1.0, x
            ra = node_id(rows, cols, i, j, a)
            rb = node_id(rows, cols, i, j, b)
            mat[node_id(rows, cols, i, j, 1)] = (1.0 - c1) * mat[ra] + c1 * mat[rb]
            mat[node_id(rows, cols, i, j, 2)] = (1.0 - c2) * mat[ra] + c2 * mat[rb]

    for side in range(4):
        for i, j in outer_boundary_linkages(rows, cols, side):
            horizontal = int(is_horizontal(i, j))
            grow = (side + 1) % 4
            if side % 2 == 1:
                coeff = 1.0 + horizontal
                other = [(grow + 2) % 4, (grow + 1) % 4]
            else:
                coeff = 2.0 - horizontal
                other = [(grow + 1) % 4, (grow + 2) % 4]
            row = node_id(rows, cols, i, j, grow)
            ra = node_id(rows, cols, i, j, other[1 - horizontal])
            rb = node_id(rows, cols, i, j, other[horizontal])
            mat[row] = (1.0 - coeff) * mat[ra] + coeff * mat[rb]
    return mat


def square_boundary(rows, cols):
    dirs = np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    corners = []
    points = []
    for side in range(4):
        for z, (i, j) in enumerate(inner_boundary_linkages(rows, cols, side)):
            parallel = parallel_to_boundary(i, j, side)
            if z == 0:
                corner = np.array([j, -i], dtype=float) + dirs[side]
                if not parallel:
                    corner += dirs[(side - 1) % 4]
                corners.append(corner)
            if not parallel:
                points.append(np.array([j, -i], dtype=float) + dirs[side])
    return np.vstack(corners), np.vstack(points)


def build_geometry_context(rows, cols):
    corners, boundary_points = square_boundary(rows, cols)
    linkages = build_linkages(rows, cols)
    quads = build_quads(rows, cols)
    linkage_to_quads = build_linkage_to_quads(linkages, quads)
    return {
        "rows": rows,
        "cols": cols,
        "corners": corners,
        "boundary_points": boundary_points,
        "linkages": linkages,
        "quads": quads,
        "linkage_to_quads": linkage_to_quads,
    }


def solve_flat_points(rows, cols, x_matrix, corners, boundary_points):
    mat = design_matrix(rows, cols, x_matrix)
    boundary_ids = []
    for side in range(4):
        boundary_ids.extend(outer_boundary_node_ids(rows, cols, side))
    a = mat[boundary_ids, : rows + cols]
    cond = np.linalg.cond(a)
    if not np.isfinite(cond) or cond > MAX_COND:
        raise np.linalg.LinAlgError(f"boundary system is ill-conditioned (cond={cond:.2e})")
    if cond < RIDGE_THRESHOLD:
        edge_seeds = np.linalg.solve(a, boundary_points)
    else:
        ata = a.T @ a
        edge_seeds = np.linalg.solve(
            ata + RIDGE_LAMBDA * np.eye(ata.shape[0]),
            a.T @ boundary_points,
        )
    return mat @ np.vstack([edge_seeds, corners])


def rotate_points(points, origin, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    return (points - origin) @ rot.T + origin


def signed_angle(a, b, c):
    u = b - a
    v = c - a
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-14 or nv < 1e-14:
        return 0.0
    u /= nu
    v /= nv
    return float(np.arctan2(u[0] * v[1] - u[1] * v[0], u @ v) % (2.0 * np.pi))


def layout_linkage(points, linkages, quads, linkage_to_quads, linkage_index, phi, phi_index):
    quad_ids = linkage_to_quads[linkage_index]
    linkage = linkages[linkage_index]
    linkage_quads = quads[quad_ids]
    order = [(phi_index + k) % 4 for k in range(4)]
    lp = np.vstack([points[linkage[k]] for k in order]).astype(float)
    qp = np.vstack([points[linkage_quads[k]] for k in order]).astype(float)
    angle = signed_angle(lp[0], lp[1], lp[-1]) - phi
    for k in range(1, 4):
        qp[k * 4 :] = rotate_points(qp[k * 4 :], lp[k - 1], angle)
        lp[k:] = rotate_points(lp[k:], lp[k - 1], angle)
        angle = -angle
    return linkage_quads, np.roll(qp, 4 * phi_index, axis=0)


def _checkerboard_phi(i, j, phi):
    return float(phi) if (i + j) % 2 == 0 else float(np.pi - phi)


def _decode_linkage_points(p0, p3, x_value, phi):
    seed = np.asarray(p3, dtype=np.float64) - np.asarray(p0, dtype=np.float64)
    c = np.cos(-float(phi))
    s = np.sin(-float(phi))
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    edge = float(x_value) * (rot @ seed)
    return np.vstack([p0, p0 + edge, p3 + edge, p3])


def _rigid_fit(src, dst):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape:
        raise ValueError(f"Rigid-fit point sets must have the same shape, got {src.shape} and {dst.shape}.")
    if src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(f"Rigid-fit expects 2D point arrays, got {src.shape}.")
    if src.shape[0] == 0:
        raise ValueError("Rigid-fit requires at least one point.")

    src_center = src.mean(axis=0)
    dst_center = dst.mean(axis=0)
    src0 = src - src_center
    dst0 = dst - dst_center
    cross = src0.T @ dst0
    u, _, vt = np.linalg.svd(cross)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0.0:
        vt[-1] *= -1.0
        rot = vt.T @ u.T
    shift = dst_center - src_center @ rot.T
    return rot, shift


def _cluster_linkage_points(linkage, quad_nodes, cluster_points):
    quad_nodes_flat = np.asarray(quad_nodes).ravel()
    linkage_points = []
    for node in linkage:
        matches = np.flatnonzero(quad_nodes_flat == node)
        if matches.size == 0:
            raise ValueError(f"Linkage node {node} not present in its local cluster.")
        linkage_points.append(cluster_points[int(matches[0])])
    return np.vstack(linkage_points)


def compute_pose_points(
    points,
    linkages,
    quads,
    linkage_to_quads,
    rows,
    cols,
    phi=0.0,
    x_matrix=None,
):
    if x_matrix is None:
        raise ValueError("compute_pose_points now requires x_matrix and no longer supports the legacy decoder.")

    x_values = np.asarray(x_matrix, dtype=np.float64)
    if x_values.shape != (rows, cols):
        raise ValueError(f"x_matrix must have shape {(rows, cols)}")

    posed_points = np.full_like(points, np.nan)

    def store(linkage_index, local_points):
        for k, node in enumerate(quads[linkage_to_quads[linkage_index]].ravel()):
            if np.any(np.isnan(posed_points[node])):
                posed_points[node] = local_points[k]

    linkage_index = 0
    for i in range(rows):
        for j in range(cols):
            linkage = linkages[linkage_index]
            p0 = (
                posed_points[linkage[0]]
                if np.all(np.isfinite(posed_points[linkage[0]]))
                else points[linkage[0]]
            )
            p3 = (
                posed_points[linkage[3]]
                if np.all(np.isfinite(posed_points[linkage[3]]))
                else points[linkage[3]]
            )
            target_linkage = _decode_linkage_points(
                p0,
                p3,
                x_values[i, j],
                _checkerboard_phi(i, j, phi),
            )

            if i == 0 and j == 0:
                local_phi = float(phi)
                phi_index = 0
            elif i == 0:
                local_phi = signed_angle(
                    posed_points[linkage[0]],
                    posed_points[linkage[1]],
                    posed_points[linkage[3]],
                )
                phi_index = 0
            else:
                local_phi = signed_angle(
                    posed_points[linkage[3]],
                    posed_points[linkage[0]],
                    posed_points[linkage[2]],
                )
                phi_index = 3

            quad_nodes, local_points = layout_linkage(
                points,
                linkages,
                quads,
                linkage_to_quads,
                linkage_index,
                local_phi,
                phi_index,
            )
            cluster_linkage = _cluster_linkage_points(linkage, quad_nodes, local_points)
            rot, shift = _rigid_fit(cluster_linkage, target_linkage)
            aligned_points = local_points @ rot.T + shift
            store(linkage_index, aligned_points)

            # Shared linkage nodes follow the direct marching decode exactly.
            for corner_idx, node in enumerate(linkage):
                posed_points[node] = target_linkage[corner_idx]

            linkage_index += 1

    return posed_points


def center_points(points, phi=None):
    pts = points.copy()
    if phi is not None:
        pts = rotate_points(pts, np.array([0.0, 0.0]), -(np.pi - phi) / 2.0)
    pts[:, 0] -= 0.5 * (pts[:, 0].max() + pts[:, 0].min())
    pts[:, 1] -= 0.5 * (pts[:, 1].max() + pts[:, 1].min())
    return pts


def rasterize(points, quads, height, width):
    xs = points[:, 0]
    ys = points[:, 1]
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    sx = (width - 1) / (xmax - xmin) if xmax > xmin else 1.0
    sy = (height - 1) / (ymax - ymin) if ymax > ymin else 1.0
    scale = min(sx, sy)

    verts = []
    codes = []
    for quad in quads:
        poly = points[np.asarray(quad, dtype=int)]
        for p in poly:
            verts.append(((p[0] - xmin) * scale, (ymax - p[1]) * scale))
        verts.append((0.0, 0.0))
        codes.extend([Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])

    xv, yv = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    grid = np.column_stack([xv.ravel(), yv.ravel()])
    mask = Path(verts, codes).contains_points(grid).reshape(height, width).astype(np.float32)
    return mask, scale


def clip_x_matrix(x_matrix, x_min=None, x_max=None):
    values = np.asarray(x_matrix, dtype=np.float64)
    clipped = values.copy()
    lower_violation = np.zeros_like(values)
    upper_violation = np.zeros_like(values)

    if x_min is not None:
        lower_violation = np.maximum(x_min - values, 0.0)
        clipped = np.maximum(clipped, x_min)
    if x_max is not None:
        upper_violation = np.maximum(values - x_max, 0.0)
        clipped = np.minimum(clipped, x_max)

    violation = lower_violation + upper_violation
    return clipped, {
        "range_violation_l1": float(np.mean(violation)) if violation.size else 0.0,
        "range_violation_max": float(np.max(violation)) if violation.size else 0.0,
        "clipped_fraction": float(np.mean(violation > 0.0)) if violation.size else 0.0,
    }


def compute_structure_points(
    rows,
    cols,
    x_matrix,
    context,
    phi=0.0,
    x_min=None,
    x_max=None,
    normalize_phi=None,
):
    x_eval, range_stats = clip_x_matrix(x_matrix, x_min=x_min, x_max=x_max)
    flat_points = solve_flat_points(rows, cols, x_eval, context["corners"], context["boundary_points"])
    points = compute_pose_points(
        flat_points,
        context["linkages"],
        context["quads"],
        context["linkage_to_quads"],
        rows,
        cols,
        phi=phi,
        x_matrix=x_eval,
    )
    if np.any(np.isnan(points)):
        raise ValueError("deployment produced NaN coordinates")
    return center_points(points, phi=normalize_phi), x_eval, range_stats


def overlap_ratio(points, quads, mask, scale):
    union_area = float(mask.sum()) / (scale * scale)
    total = 0.0
    for quad in quads:
        poly = points[np.asarray(quad, dtype=int)]
        total += 0.5 * abs(
            sum(
                poly[k, 0] * poly[(k + 1) % 4, 1]
                - poly[(k + 1) % 4, 0] * poly[k, 1]
                for k in range(4)
            )
        )
    if total <= 1e-12:
        return 1.0
    return max(0.0, min(1.0, 1.0 - union_area / total))


def quad_is_valid(poly):
    if not np.all(np.isfinite(poly)):
        return False
    area = 0.5 * sum(
        poly[k, 0] * poly[(k + 1) % 4, 1] - poly[(k + 1) % 4, 0] * poly[k, 1]
        for k in range(4)
    )
    if abs(area) < 1e-10:
        return False
    return not (
        segments_intersect(poly[0], poly[1], poly[2], poly[3])
        or segments_intersect(poly[1], poly[2], poly[3], poly[0])
    )


def segments_intersect(a, b, c, d):
    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]

    r = b - a
    s = d - c
    denom = cross(r, s)
    if abs(denom) < 1e-12:
        return False
    t = cross(c - a, s) / denom
    u = cross(c - a, r) / denom
    return 0.0 < t < 1.0 and 0.0 < u < 1.0


def render_structure_mask_and_metrics(
    rows,
    cols,
    x_matrix,
    context,
    height,
    width,
    x_min=None,
    x_max=None,
):
    clipped, range_stats = clip_x_matrix(x_matrix, x_min=x_min, x_max=x_max)
    metrics = {
        "ok": False,
        "invalid_quad_count": 0,
        "overlap_ratio": 1.0,
        "fill_ratio": 0.0,
        "range_violation_l1": range_stats["range_violation_l1"],
        "range_violation_max": range_stats["range_violation_max"],
        "clipped_fraction": range_stats["clipped_fraction"],
        "error": None,
    }

    try:
        posed_points, clipped, _ = compute_structure_points(
            rows,
            cols,
            clipped,
            context,
            phi=0.0,
            normalize_phi=None,
        )
        invalid_count = sum(
            not quad_is_valid(posed_points[np.asarray(quad, dtype=int)]) for quad in context["quads"]
        )
        mask, scale = rasterize(posed_points, context["quads"], height, width)
        metrics.update(
            {
                "ok": True,
                "invalid_quad_count": int(invalid_count),
                "overlap_ratio": float(overlap_ratio(posed_points, context["quads"], mask, scale)),
                "fill_ratio": float(mask.mean()),
            }
        )
        return mask, metrics, posed_points, clipped
    except Exception as exc:
        metrics["error"] = str(exc)
        return np.zeros((height, width), dtype=np.float32), metrics, None, clipped


def mask_iou(pred_mask, gt_mask, threshold=0.5):
    pred = np.asarray(pred_mask, dtype=np.float32) >= threshold
    gt = np.asarray(gt_mask, dtype=np.float32) >= threshold
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(pred, gt).sum() / union)


def _mask_similarity_stats(mask):
    ys, xs = np.nonzero(mask)
    area = int(xs.size)
    if area == 0:
        return None

    coords = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    center = coords.mean(axis=0)
    centered = coords - center
    if area >= 2:
        cov = (centered.T @ centered) / float(area)
        evals, evecs = np.linalg.eigh(cov)
        major = evecs[:, int(np.argmax(evals))]
        angle = math.atan2(float(major[1]), float(major[0]))
        major_val = max(float(evals.max()), 0.0)
        minor_val = max(float(evals.min()), 0.0)
        anisotropy = (major_val + 1e-6) / (minor_val + 1e-6)
    else:
        angle = 0.0
        anisotropy = 1.0

    return {
        "area": float(area),
        "center": center,
        "angle": angle,
        "anisotropy": anisotropy,
    }


def _warp_mask_similarity(mask, *, scale, angle, src_center, dst_center, output_shape):
    height, width = output_shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    x_rel = xx - float(dst_center[0])
    y_rel = yy - float(dst_center[1])

    c = math.cos(float(angle))
    s = math.sin(float(angle))
    safe_scale = max(float(scale), 1e-6)
    x_in = (c * x_rel + s * y_rel) / safe_scale + float(src_center[0])
    y_in = (-s * x_rel + c * y_rel) / safe_scale + float(src_center[1])

    xi = np.rint(x_in).astype(np.int64)
    yi = np.rint(y_in).astype(np.int64)
    inside = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)

    warped = np.zeros((height, width), dtype=bool)
    warped[inside] = mask[yi[inside], xi[inside]]
    return warped


def mask_siou(
    pred_mask,
    gt_mask,
    threshold=0.5,
    *,
    isotropy_ratio_threshold=1.15,
    coarse_angle_steps=24,
):
    pred = np.asarray(pred_mask, dtype=np.float32) >= threshold
    gt = np.asarray(gt_mask, dtype=np.float32) >= threshold

    pred_stats = _mask_similarity_stats(pred)
    gt_stats = _mask_similarity_stats(gt)
    if pred_stats is None or gt_stats is None:
        return 0.0

    scale = math.sqrt(pred_stats["area"] / max(gt_stats["area"], 1e-6))
    base_angle = float(pred_stats["angle"] - gt_stats["angle"])
    candidates = [base_angle, base_angle + math.pi]
    if (
        pred_stats["anisotropy"] < float(isotropy_ratio_threshold)
        or gt_stats["anisotropy"] < float(isotropy_ratio_threshold)
    ):
        candidates.extend((2.0 * math.pi * k) / int(coarse_angle_steps) for k in range(int(coarse_angle_steps)))

    unique = []
    seen = set()
    period = 2.0 * math.pi
    for angle in candidates:
        key = round(float(angle % period), 6)
        if key in seen:
            continue
        seen.add(key)
        unique.append(float(angle))

    best = 0.0
    for angle in unique:
        aligned_gt = _warp_mask_similarity(
            gt,
            scale=scale,
            angle=angle,
            src_center=gt_stats["center"],
            dst_center=pred_stats["center"],
            output_shape=pred.shape,
        )
        union = np.logical_or(pred, aligned_gt).sum()
        if union == 0:
            continue
        score = float(np.logical_and(pred, aligned_gt).sum() / union)
        if score > best:
            best = score
    return best


def mask_overlay_rgb(pred_mask, gt_mask, threshold=0.5):
    pred = np.asarray(pred_mask, dtype=np.float32) >= threshold
    gt = np.asarray(gt_mask, dtype=np.float32) >= threshold
    overlay = np.zeros(pred.shape + (3,), dtype=np.float32)
    overlay[np.logical_and(pred, gt)] = (0.0, 1.0, 0.0)
    overlay[np.logical_and(pred, np.logical_not(gt))] = (1.0, 0.0, 0.0)
    overlay[np.logical_and(np.logical_not(pred), gt)] = (0.0, 0.0, 1.0)
    return overlay


def build_dataset_entry(rows, cols, x_matrix, context, height, width):
    mask, metrics, _, clipped = render_structure_mask_and_metrics(
        rows,
        cols,
        x_matrix,
        context,
        height,
        width,
    )
    if not metrics["ok"] or metrics["invalid_quad_count"] > 0:
        return None
    if metrics["fill_ratio"] < MASK_MIN_FILL or metrics["fill_ratio"] > MASK_MAX_FILL:
        return None
    if metrics["overlap_ratio"] > MAX_OVERLAP:
        return None

    return {
        "image": clipped.astype(np.float32)[None, :, :],
        "mask": mask[None, :, :],
        "metadata": {
            "grid_rows": rows,
            "grid_cols": cols,
            "phi_mask": 0.0,
            "x_matrix": clipped.astype(np.float32),
        },
    }
