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


def build_context(rows, cols):
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


def solve_points(rows, cols, x_matrix, corners, boundary_points):
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


def best_fit(placed, points, quad_nodes):
    src = []
    dst = []
    seen = set()
    for k, node in enumerate(np.asarray(quad_nodes).ravel()):
        if node in seen or np.any(np.isnan(placed[node])):
            continue
        seen.add(node)
        src.append(points[k])
        dst.append(placed[node])
    if not src:
        return points
    src = np.asarray(src)
    dst = np.asarray(dst)
    if len(src) == 1:
        return points + (dst[0] - src[0])

    best = (0, 1, -1.0)
    for i in range(len(src)):
        for j in range(i + 1, len(src)):
            dist = float(np.linalg.norm(dst[j] - dst[i]))
            if dist > best[2]:
                best = (i, j, dist)
    i, j, _ = best
    va = src[j] - src[i]
    vb = dst[j] - dst[i]
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < 1e-12 or nb < 1e-12:
        return points + (dst[0] - src[0])
    va /= na
    vb /= nb
    c = float(va @ vb)
    s = float(va[0] * vb[1] - va[1] * vb[0])
    rot = np.array([[c, -s], [s, c]])
    shift = dst[i] - rot @ src[i]
    return points @ rot.T + shift


def deploy(points, linkages, quads, linkage_to_quads, rows, cols, phi=0.0):
    deployed = np.full_like(points, np.nan)

    def store(linkage_index, local_points):
        for k, node in enumerate(quads[linkage_to_quads[linkage_index]].ravel()):
            if np.any(np.isnan(deployed[node])):
                deployed[node] = local_points[k]

    _, first = layout_linkage(points, linkages, quads, linkage_to_quads, 0, phi, 0)
    store(0, first)

    linkage_index = 1
    for i in range(rows):
        for j in range(cols):
            if i == 0 and j == 0:
                continue
            linkage = linkages[linkage_index]
            if i == 0:
                sub_phi = signed_angle(
                    deployed[linkage[0]],
                    deployed[linkage[1]],
                    deployed[linkage[3]],
                )
                quad_nodes, local_points = layout_linkage(
                    points,
                    linkages,
                    quads,
                    linkage_to_quads,
                    linkage_index,
                    sub_phi,
                    0,
                )
            else:
                sub_phi = signed_angle(
                    deployed[linkage[3]],
                    deployed[linkage[0]],
                    deployed[linkage[2]],
                )
                quad_nodes, local_points = layout_linkage(
                    points,
                    linkages,
                    quads,
                    linkage_to_quads,
                    linkage_index,
                    sub_phi,
                    3,
                )
            store(linkage_index, best_fit(deployed, local_points, quad_nodes))
            linkage_index += 1
    return deployed


def normalize_points(points, phi=None):
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


def make_sample(rows, cols, x_matrix, context, height, width):
    try:
        flat_points = solve_points(
            rows,
            cols,
            x_matrix,
            context["corners"],
            context["boundary_points"],
        )
    except:
        return None

    deployed = deploy(
        flat_points,
        context["linkages"],
        context["quads"],
        context["linkage_to_quads"],
        rows,
        cols,
        phi=0.0,
    )
    if np.any(np.isnan(deployed)):
        return None
    if any(not quad_is_valid(deployed[np.asarray(quad, dtype=int)]) for quad in context["quads"]):
        return None

    deployed = normalize_points(deployed)
    mask, scale = rasterize(deployed, context["quads"], height, width)
    fill = float(mask.mean())
    if fill < MASK_MIN_FILL or fill > MASK_MAX_FILL:
        return None
    if overlap_ratio(deployed, context["quads"], mask, scale) > MAX_OVERLAP:
        return None

    return {
        "image": x_matrix.astype(np.float32)[None, :, :],
        "mask": mask[None, :, :],
        "metadata": {
            "grid_rows": rows,
            "grid_cols": cols,
            "phi_mask": 0.0,
            "x_matrix": x_matrix.astype(np.float32),
        },
    }
