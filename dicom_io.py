"""极简 DICOM 读取器：只依赖 numpy。支持显式/隐式 VR、小端、未压缩像素数据。"""
import struct, glob, os
import numpy as np

VR_SHORT = {'AE','AS','AT','CS','DA','DS','DT','FL','FD','IS','LO','LT','PN','SH','SL','SS','ST','TM','UI','UL','US','SV','UV'}
VR_LONG  = {'OB','OD','OF','OL','OV','OW','SQ','UC','UN','UR','UT'}


def _elem(b, off, explicit=True):
    tag = struct.unpack_from('<HH', b, off); off += 4
    if explicit:
        vr = b[off:off+2].decode('ascii', 'replace'); off += 2
        if vr in VR_LONG:
            off += 2
            ln = struct.unpack_from('<I', b, off)[0]; off += 4
        elif vr in VR_SHORT:
            ln = struct.unpack_from('<H', b, off)[0]; off += 2
        else:
            off -= 2
            ln = struct.unpack_from('<I', b, off)[0]; off += 4
    else:
        vr = None
        ln = struct.unpack_from('<I', b, off)[0]; off += 4
    if ln == 0xFFFFFFFF:
        return tag, vr, None, None
    if off + ln > len(b):
        ln = len(b) - off
    return tag, vr, b[off:off+ln], off + ln


def read_file(path):
    with open(path, 'rb') as f:
        b = f.read()
    off = 132 if b[128:132] == b'DICM' else 0
    meta = {}
    if off:
        while off < len(b) - 8:
            tag, vr, val, nxt = _elem(b, off, True)
            if tag[0] != 2 or val is None: break
            meta[tag] = val; off = nxt
        glen = struct.unpack('<I', meta.get((2, 0), b'\0\0\0\0'))[0]
        if glen: off = 132 + 12 + glen
    ts = meta.get((2, 0x10), b'1.2.840.10008.1.2.1').decode('ascii', 'replace').strip('\x00 ')
    explicit = ts != '1.2.840.10008.1.2'
    hdr = {}
    while off < len(b) - 8:
        tag, vr, val, nxt = _elem(b, off, explicit)
        if val is None: break
        if tag == (0x7FE0, 0x0010):
            hdr[tag] = (vr, None, off, len(b) - off)
            break
        hdr[tag] = (vr, val, off, len(val)); off = nxt
    return b, hdr


def get_str(hdr, tag, default=''):
    e = hdr.get(tag)
    if not e or e[1] is None: return default
    return e[1].decode('utf-8', 'replace').strip('\x00 ')


def get_num(hdr, tag, default=None):
    e = hdr.get(tag)
    if not e: return default
    vr, val = e[0], e[1]
    if val is None: return default
    try:
        if vr == 'US': return float(struct.unpack('<H', val[:2])[0])
        if vr == 'SS': return float(struct.unpack('<h', val[:2])[0])
        if vr == 'UL': return float(struct.unpack('<I', val[:4])[0])
        if vr == 'SL': return float(struct.unpack('<i', val[:4])[0])
        if vr == 'FL': return float(struct.unpack('<f', val[:4])[0])
        if vr == 'FD': return float(struct.unpack('<d', val[:8])[0])
        return float(val.decode('ascii', 'replace').strip('\x00 ').split('\\')[0])
    except Exception:
        return default


def get_nums(hdr, tag, default=None):
    e = hdr.get(tag)
    if not e or e[1] is None: return default
    try:
        return [float(x) for x in e[1].decode('ascii', 'replace').strip('\x00 ').split('\\')]
    except Exception:
        return default


def load_series(folder, progress=None):
    files = sorted(glob.glob(os.path.join(folder, '*.dcm'))) or \
            sorted(glob.glob(os.path.join(folder, '*.DCM'))) or \
            sorted(glob.glob(os.path.join(folder, '*')))
    files = [f for f in files if os.path.isfile(f) and not os.path.basename(f).startswith('.')]
    recs = []
    for i, p in enumerate(files):
        b, h = read_file(p)
        rows = int(get_num(h, (0x28, 0x10), 0) or 0)
        cols = int(get_num(h, (0x28, 0x11), 0) or 0)
        if not rows or not cols:
            continue
        slope = get_num(h, (0x28, 0x1053), 1.0) or 1.0
        inter = get_num(h, (0x28, 0x1052), 0.0) or 0.0
        ipp = get_nums(h, (0x20, 0x32), [0, 0, i]) or [0, 0, i]
        off = h[(0x7FE0, 0x10)][2]
        n = rows * cols
        if len(b) - off >= n * 2:
            arr = np.frombuffer(b, '<i2', n, off).reshape(rows, cols).astype(np.float32)
        else:
            arr = np.frombuffer(b, np.uint8, n, off).reshape(rows, cols).astype(np.float32)
        recs.append(dict(path=p, ipp=ipp, inst=get_num(h, (0x20, 0x13), i),
                         arr=arr * slope + inter, rows=rows, cols=cols,
                         sp=get_nums(h, (0x28, 0x30), [1, 1]) or [1, 1],
                         th=get_num(h, (0x18, 0x50), 1.0) or 1.0,
                         hdr=h, b=b))
        if progress and i % 50 == 0:
            progress(i, len(files))
        del b
    recs.sort(key=lambda r: r['ipp'][2])
    vol = np.stack([r['arr'] for r in recs], 0)      # (z, y, x)
    hdr0 = recs[0]['hdr']

    # ---- 从 DICOM 方向字段推导屏幕四边的解剖方位（不靠猜） ----
    # IOP 前 3 个数 = 像素"列"增大方向（屏幕从左往右）在人体 LPS 坐标中的指向
    # IOP 后 3 个数 = 像素"行"增大方向（屏幕从上往下）在人体 LPS 坐标中的指向
    # LPS: +x = 患者左(Left)  +y = 后(Posterior)  +z = 上(Superior)
    iop = get_nums(hdr0, (0x20, 0x37), [1, 0, 0, 0, 1, 0]) or [1, 0, 0, 0, 1, 0]
    dirX = np.array(iop[0:3], float)                 # 体数据 x 增大 → 人体方向
    dirY = np.array(iop[3:6], float)                 # 体数据 y 增大 → 人体方向
    dirZ = np.cross(dirX, dirY)                      # 体数据 z 增大 → 人体方向
    for d in (dirX, dirY, dirZ):
        n = np.linalg.norm(d)
        if n > 0:
            d /= n
    orient = dict(
        x_to_right=bool(dirX[0] < 0),   # True = 体数据 x 增大是往患者右边
        y_to_post=bool(dirY[1] > 0),    # True = 体数据 y 增大是往后
        z_to_sup=bool(dirZ[2] > 0),     # True = 体数据 z 增大是往上
        iop=[float(v) for v in iop],
    )
    info = dict(
        n=len(recs), shape=list(vol.shape),
        spacing=[recs[0]['sp'][1], recs[0]['sp'][0], recs[0]['th']],
        modality=get_str(hdr0, (0x8, 0x60)),
        manufacturer=get_str(hdr0, (0x8, 0x70)),
        model=get_str(hdr0, (0x8, 0x1090)),
        bodypart=get_str(hdr0, (0x18, 0x15)),
        studydate=get_str(hdr0, (0x8, 0x20)),
        seriesdesc=get_str(hdr0, (0x8, 0x103e)),
        kvp=get_num(hdr0, (0x18, 0x60)),
        institute=get_str(hdr0, (0x8, 0x80)),
        patient_age=get_str(hdr0, (0x10, 0x1010)),
        patient_sex=get_str(hdr0, (0x10, 0x40)),
        zrange=[recs[0]['ipp'][2], recs[-1]['ipp'][2]],
        orient=orient,
    )
    return vol.astype(np.int16), info
