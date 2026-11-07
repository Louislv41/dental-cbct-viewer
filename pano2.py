import numpy as np, pano

def build_curve(mask, cy, cx, th, frac=0.45):
    th_, r, inn, out = pano.arch_curve(mask, cy, cx, nth=len(th), rmax=320, dr=0.5, frac=frac)
    valid = r > 0
    idx = np.nonzero(valid)[0]
    r2 = r.copy()
    # fill invalid angles with the outer bony contour (ramus) so the strip stays continuous
    r2[~valid] = out[~valid]
    # bridge across the gap linearly
    if (~valid).any():
        r2[~valid] = np.interp(idx[np.argsort(idx)], np.arange(len(r2)), r2)[~valid] if False else r2[~valid]
    r2 = pano.smooth_periodic(r2, 11)
    return r2

def panoramic(vol, r, cy, cx, nth=1440, half=9, ncol=1400, pct=92, center_anterior=True):
    th = np.deg2rad(np.arange(0, 360, 360.0/nth))
    r_old_g = np.linspace(0, 360, len(r), endpoint=False)
    r = np.interp(np.rad2deg(th), r_old_g, r)
    thc = np.append(th, th[0]+2*np.pi); rc = np.append(r, r[0])
    x = cx + rc*np.cos(thc); y = cy + rc*np.sin(thc)
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    # resample uniformly along arc length, starting at the most anterior point
    if center_anterior:
        ymin = np.argmin(y); s0 = s[ymin]
        su = (np.linspace(s0 - s[-1]/2, s0 + s[-1]/2, ncol) % s[-1])
    else:
        su = np.linspace(0, s[-1], ncol, endpoint=False)
    xi = np.interp(su, s, x); yi = np.interp(su, s, y)
    # local tangents -> normals
    dx = np.gradient(xi); dy = np.gradient(yi)
    nn = np.hypot(dx, dy) + 1e-9
    nx, ny = -dy/nn, dx/nn
    t = np.linspace(-1, 1, half*2+1)
    nz = vol.shape[0]; H, W = vol.shape[1:]
    out = np.zeros((nz, ncol), np.float32)
    for i in range(ncol):
        xs = np.clip((xi[i] + t*half*nx[i]).round().astype(int), 0, W-1)
        ys = np.clip((yi[i] + t*half*ny[i]).round().astype(int), 0, H-1)
        out[:, i] = np.percentile(vol[:, ys, xs], pct, axis=1)
    return out, (xi, yi)
