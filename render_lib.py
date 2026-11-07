"""渲染工具：窗宽窗位、MPR 切片、体渲染。供 app.py 与 ctlab.py 共用。"""
import io
import numpy as np
import scipy.ndimage as ndi
from PIL import Image


def window_to_u8(sl, w, l):
    lo = l - w / 2.0
    hi = l + w / 2.0
    return np.clip((sl - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)


def to_image(sl, w=3000, l=800):
    return Image.fromarray(window_to_u8(sl, w, l))


def downsample(vol, f):
    if f <= 1:
        return vol.astype(np.float32)
    Z, Y, X = (vol.shape[i] // f * f for i in range(3))
    v = vol[:Z, :Y, :X]
    return v.reshape(Z//f, f, Y//f, f, X//f, f).mean(axis=(1, 3, 5)).astype(np.float32)


def volume_render(vol, az=0.0, el=0.0, th=250.0, op=1.0, lo=250.0, hi=2600.0,
                  gamma=1.5, shade=0.75, tone_lo=200.0, tone_hi=2200.0, flip=True):
    """体渲染：沿 +y 方向（从正前方）观察，az 绕垂直轴、el 俯仰。返回 0~1 灰度图。"""
    v = vol if vol.dtype == np.float32 else vol.astype(np.float32)
    if el:
        v = ndi.rotate(v, el, axes=(0, 1), reshape=False, order=1, mode='constant', cval=-1000)
    if az:
        v = ndi.rotate(v, az, axes=(1, 2), reshape=False, order=1, mode='constant', cval=-1000)

    u = (v - lo) / max(hi - lo, 1)
    np.clip(u, 0, 1, out=u)
    a = u * u * (3.0 - 2.0 * u)
    a *= min(max(op, 0.0), 1.0) * 0.85
    u2 = (v - tone_lo) / max(tone_hi - tone_lo, 1)
    np.clip(u2, 0, 1, out=u2)
    t = u2 * (1.55 - 0.55 * u2)
    if shade > 0:
        gy = np.gradient(v, axis=1)
        gn = np.clip(-gy, 0, None)
        sc = np.percentile(np.abs(gy[::4, ::4, ::4]), 99.5) + 1e-6
        gn *= (1.0 / sc)
        t = t * (1 - shade) + t * shade * (0.5 + 0.5 * np.clip(gn, 0, 1))

    trans = np.cumprod(1.0 - a, axis=1)
    alive = (trans > 0.004).any(axis=(0, 2))
    ymax = a.shape[1] if bool(alive.all()) else int(np.argmin(alive))
    ymax = min(max(ymax, 1), a.shape[1])
    a, t, trans = a[:, :ymax, :], t[:, :ymax, :], trans[:, :ymax, :]
    T = np.empty_like(a)
    T[:, 0, :] = 1.0
    T[:, 1:, :] = trans[:, :-1, :]
    img = np.clip((T * a * t).sum(axis=1), 0, 1)
    return np.flipud(img) if flip else img


def get_slice(vol, axis, idx):
    """axis: 'z'(轴位) / 'y'(冠状位) / 'x'(矢状位)，已按阅片方向翻转。"""
    if axis == 'z':
        return vol[idx, :, :]
    if axis == 'y':
        return np.flipud(vol[:, idx, :])
    return np.flipud(vol[:, :, idx])
