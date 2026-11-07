#!/usr/bin/env python3
"""CT / CBCT 本地阅片器 —— 零依赖轻量后台（numpy + scipy + Pillow）。

用法:  python3 app.py [DICOM目录] [端口]
"""
import io, os, sys, json, time, threading, webbrowser, mimetypes
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np
from PIL import Image
import scipy.ndimage as ndi

import dicom_io

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
VOL = None
INFO = {}
LOCK = threading.Lock()
VR_CACHE = {}
_CACHE_VOL = {}
VOLC = None       # 低分辨率 float32 副本，供 3D 实时渲染


from render_lib import window_to_u8, volume_render as _vr


def png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG', compress_level=1)
    return buf.getvalue()


def jpg_bytes(arr, q=92):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='JPEG', quality=q, optimize=False, subsampling=0)
    return buf.getvalue()


def get_slice(axis, idx):
    with LOCK:
        v = VOL
        if axis == 'z':   return v[idx, :, :]
        if axis == 'y':   return np.flipud(v[:, idx, :])
        return np.flipud(v[:, :, idx])


def lowres(factor):
    key = factor
    if key in _CACHE_VOL:
        return _CACHE_VOL[key]
    v = VOL.astype(np.float32)
    if factor > 1:
        Z, Y, X = (v.shape[i] // factor * factor for i in range(3))
        v = v[:Z, :Y, :X].reshape(Z//factor, factor, Y//factor, factor, X//factor, factor).mean(axis=(1,3,5))
    _CACHE_VOL[key] = v
    return v


def volume_render(az=0.0, el=0.0, th=250.0, op=1.0, lo=250.0, hi=2600.0, factor=2,
                  gamma=1.5, shade=0.75, tone_lo=200.0, tone_hi=2200.0):
    key = (round(az, 1), round(el, 1), round(th, 0), round(op, 2),
           round(lo, 0), round(hi, 0), factor, round(shade, 2))
    with LOCK:
        if key in VR_CACHE:
            return VR_CACHE[key]
        v = lowres(factor).copy()
    img = _vr(v, az=az, el=el, th=th, op=op, lo=lo, hi=hi,
              gamma=gamma, shade=shade, tone_lo=tone_lo, tone_hi=tone_hi)
    data = png_bytes((img * 255).astype(np.uint8))
    with LOCK:
        if len(VR_CACHE) > 300:
            VR_CACHE.clear()
        VR_CACHE[key] = data
    return data


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):
        pass

    def _send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        p = u.path
        try:
            if p == '/' or p == '/index.html':
                return self._static('index.html')
            if p.startswith('/static/'):
                return self._static(p[8:])
            if p == '/api/info':
                return self._send(json.dumps(INFO).encode(), 'application/json')
            if p == '/api/slice':
                axis = q.get('axis', ['z'])[0]
                idx = int(q.get('idx', ['0'])[0])
                if q.get('raw', ['0'])[0] == '1':
                    n = INFO['shape'][{'z':0,'y':1,'x':2}[axis]]
                    idx = max(0, min(idx, n-1))
                    sl = np.ascontiguousarray(get_slice(axis, idx), dtype='<i2')
                    h, w = sl.shape
                    body = sl.tobytes()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/octet-stream')
                    self.send_header('Content-Length', str(len(body)))
                    self.send_header('X-Dims', f'{w},{h}')
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    return self.wfile.write(body)
                w = float(q.get('w', ['3000'])[0])
                l = float(q.get('l', ['800'])[0])
                n = INFO['shape'][{'z':0,'y':1,'x':2}[axis]]
                idx = max(0, min(idx, n-1))
                sl = get_slice(axis, idx)
                fmt = q.get('fmt', ['png'])[0]
                body = jpg_bytes(window_to_u8(sl, w, l), 94) if fmt == 'jpg' else png_bytes(window_to_u8(sl, w, l))
                return self._send(body, 'image/jpeg' if fmt == 'jpg' else 'image/png')
            if p == '/api/vr':
                g = lambda k, d: float(q.get(k, [d])[0])
                body = volume_render(
                    az=g('az', 0), el=g('el', 0), th=g('th', 250), op=g('op', 1.0),
                    lo=g('lo', 250), hi=g('hi', 2600), factor=int(g('q', 2)),
                    gamma=g('gamma', 1.5), shade=g('shade', 0.75))
                return self._send(body, 'image/png')
            if p == '/api/hist':
                with LOCK:
                    v = VOL[::4, ::4, ::4]
                h, edges = np.histogram(np.clip(v, -1200, 4000), bins=120, range=(-1200, 4000))
                return self._send(json.dumps({'h': h.tolist(), 'e': edges.tolist()}).encode(), 'application/json')
            self._send(b'not found', 'text/plain', 404)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._send(str(e).encode(), 'text/plain', 500)

    def _static(self, name):
        fp = os.path.join(HERE, 'static', name)
        if not os.path.isfile(fp):
            return self._send(b'not found', 'text/plain', 404)
        ctype = mimetypes.guess_type(fp)[0] or 'application/octet-stream'
        with open(fp, 'rb') as f:
            data = f.read()
        self._send(data, ctype + ('; charset=utf-8' if ctype.startswith('text/') else ''))


def main():
    global VOL, INFO, VOLC
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    if not folder:
        here = os.path.dirname(HERE)
        cands = [os.path.join(here, d) for d in os.listdir(here)
                 if os.path.isdir(os.path.join(here, d)) and
                 any(f.lower().endswith('.dcm') for f in os.listdir(os.path.join(here, d)))]
        folder = cands[0] if cands else here
    print(f'读取 DICOM: {folder}', flush=True)
    t = time.time()
    VOL, INFO = dicom_io.load_series(folder, progress=lambda i, n: print(f'  {i}/{n}', end='\r', flush=True))
    print(f'\n完成 {VOL.shape}  用时 {time.time()-t:.1f}s', flush=True)
    INFO['spacing'] = INFO['spacing']
    INFO['source'] = os.path.basename(folder.rstrip('/'))
    INFO['load_seconds'] = round(time.time() - t, 2)
    lowres(2); lowres(3)
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8788
    for attempt in range(20):
        try:
            srv = ThreadingHTTPServer(('127.0.0.1', port), Handler)
            break
        except OSError:
            port += 1
    else:
        raise SystemExit('端口都被占用了')
    url = f'http://127.0.0.1:{port}/'
    print(f'\n  ✓ 阅片器已启动 → {url}')
    print('    浏览器会自动打开；如果没打开，手动复制上面的地址\n'
          '    关闭这个窗口即退出\n', flush=True)
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('bye')


if __name__ == '__main__':
    main()
