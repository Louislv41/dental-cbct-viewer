import numpy as np, scipy.ndimage as ndi

def radial_profile(mask, cy, cx, nth=720, rmax=300, dr=0.5):
    th = np.deg2rad(np.arange(0, 360, 360.0/nth))
    rr = np.arange(0, rmax, dr)
    R, T = np.meshgrid(rr, th, indexing='ij')          # (nr, nth)
    yy = (cy + R*np.sin(T)).round().astype(int)
    xx = (cx + R*np.cos(T)).round().astype(int)
    ok = (yy >= 0) & (yy < mask.shape[0]) & (xx >= 0) & (xx < mask.shape[1])
    prof = np.zeros(R.shape, np.float32)
    prof[ok] = mask[yy[ok], xx[ok]]
    return th, rr, prof

def arch_curve(mask, cy, cx, nth=720, rmax=300, dr=0.5, frac=0.45):
    th, rr, prof = radial_profile(mask, cy, cx, nth, rmax, dr)
    has = prof > 0.5
    out = np.zeros(len(th)); inn = np.zeros(len(th))
    for i in range(len(th)):
        idx = np.nonzero(has[:, i])[0]
        if len(idx): inn[i], out[i] = rr[idx[0]], rr[idx[-1]]
    r = inn + frac*np.maximum(out-inn, 0)
    return th, r, inn, out

def smooth_periodic(x, win=15):
    k = np.ones(win)/win
    ext = np.concatenate([x[-win:], x, x[:win]])
    return np.convolve(ext, k, mode='same')[win:-win]

def panorama(vol, th, r, cy, cx, zaxis=0, half=14, nsample_z=None):
    """Curved planar reformation: (theta along x, z along y)."""
    nz, H, W = vol.shape
    nth = len(th)
    zz = np.arange(nz)
    out = np.zeros((nz, nth), np.float32)
    t = np.linspace(0, 1, half*2+1) - 0.5
    for i in range(nth):
        # local radial direction
        dx, dy = np.cos(th[i]), np.sin(th[i])
        xs = cx + (r[i] + t*half)*dx
        ys = cy + (r[i] + t*half)*dy
        xi = np.clip(xs.round().astype(int), 0, W-1)
        yi = np.clip(ys.round().astype(int), 0, H-1)
        vol_slab = vol[zz][:, yi, xi]          # (nz, nsamples)
        out[:, i] = np.percentile(vol_slab, 92, axis=1)
    return out
