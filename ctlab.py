#!/usr/bin/env python3
"""ctlab —— 牙科/医学 CT、CBCT 命令行工具。

常用命令
--------
  python3 ctlab.py info                                 查看检查信息
  python3 ctlab.py slice --axis z --at 200 -o a.png     导出单张切面
  python3 ctlab.py slice --axis y --range 100:300:20 -o out/   批量导出
  python3 ctlab.py mpr -o out/                          轴位/冠状位/矢状位三套序列
  python3 ctlab.py pano -o 全景片.png                    生成全景片
  python3 ctlab.py render3d --az 30 --el 10 -o 3d.png    生成三维图
  python3 ctlab.py spin -o 旋转.mp4                      生成 3D 旋转视频
  python3 ctlab.py montage --axis z --step 10 -o m.jpg   拼接成多格大图

参数
----
  --axis   z=轴位(横断面)  y=冠状位  x=矢状位
  --at     第几层（从 0 开始）
  --range  start:stop:step
  --w --l  窗宽 / 窗位（默认 3000 / 800）
  --az --el  3D 的方位角 / 俯仰角
  --th     3D 阈值（骨 300，只看牙齿 900）
  --op     3D 不透明度 0~1
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

import dicom_io
import render_lib as rl


def auto_src():
    """没指定 --src 时，自动找本目录或上一级目录里含 .dcm 的文件夹。"""
    for base in ('.', '..'):
        try:
            for d in sorted(os.listdir(base)):
                full = os.path.join(base, d)
                if os.path.isdir(full) and any(
                        f.lower().endswith('.dcm') for f in os.listdir(full)):
                    return full
        except OSError:
            pass
    return '.'


def find_series(path):
    if not path:
        path = auto_src()
    if os.path.isfile(path):
        path = os.path.dirname(path)
    return path


def load(path):
    folder = find_series(path)
    print(f'读取 {folder} …', file=sys.stderr)
    vol, info = dicom_io.load_series(folder)
    print(f'  体积 {vol.shape}  体素 {info["spacing"]} mm', file=sys.stderr)
    return vol, info


def save(img, out):
    d = os.path.dirname(os.path.abspath(out))
    os.makedirs(d, exist_ok=True)
    img.save(out)
    print(f'  → {out}')


def parse_range(s, n):
    if s is None:
        return None
    parts = [int(x) if x else None for x in s.replace(':', ' ').split()]
    if len(parts) == 1:
        return [parts[0]]
    a = parts[0] or 0
    b = parts[1] if len(parts) > 1 and parts[1] else n
    st = parts[2] if len(parts) > 2 and parts[2] else 1
    return list(range(a, min(b, n), st))


# ---------------------------------------------------------------- commands
def cmd_info(args):
    vol, info = load(args.src)
    for k, v in info.items():
        print(f'{k:14s}: {v}')


def cmd_slice(args):
    vol, info = load(args.src)
    axis = args.axis
    n = {'z': vol.shape[0], 'y': vol.shape[1], 'x': vol.shape[2]}[axis]
    idxs = parse_range(args.range, n) if args.range else [args.at]
    one = len(idxs) == 1 and not args.out.endswith(os.sep)
    for i in idxs:
        i = max(0, min(i, n - 1))
        im = rl.to_image(rl.get_slice(vol, axis, i), args.w, args.l)
        out = args.out
        if not one:
            os.makedirs(out, exist_ok=True)
            out = os.path.join(out, f'{axis}_{i:04d}.png')
        save(im, out)


def cmd_mpr(args):
    vol, info = load(args.src)
    for axis, name in (('z', 'axial'), ('y', 'coronal'), ('x', 'sagittal')):
        n = {'z': vol.shape[0], 'y': vol.shape[1], 'x': vol.shape[2]}[axis]
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        for i in range(0, n, args.step):
            im = rl.to_image(rl.get_slice(vol, axis, i), args.w, args.l)
            im.save(os.path.join(d, f'{name}_{i:04d}.png'))
        print(f'  → {d}  ({len(range(0,n,args.step))} 张)')


def cmd_montage(args):
    vol, info = load(args.src)
    axis = args.axis
    n = {'z': vol.shape[0], 'y': vol.shape[1], 'x': vol.shape[2]}[axis]
    idxs = parse_range(args.range, n) if args.range else list(range(0, n, args.step))
    tiles = []
    for i in idxs:
        im = rl.to_image(rl.get_slice(vol, axis, i), args.w, args.l)
        im = im.resize((args.tile, args.tile), Image.LANCZOS).convert('RGB')
        tiles.append(im)
    cols = args.cols
    rows = int(np.ceil(len(tiles) / cols))
    pad = 4
    M = Image.new('RGB', (cols * (args.tile + pad) + pad, rows * (args.tile + pad) + pad), (0, 0, 0))
    for k, im in enumerate(tiles):
        r, c = divmod(k, cols)
        M.paste(im, (pad + c * (args.tile + pad), pad + r * (args.tile + pad)))
    save(M, args.out)


def cmd_pano(args):
    import pano2
    vol, info = load(args.src)
    mask = vol[max(0, args.level - 25):args.level + 25].max(axis=0) > 1800
    r = pano2.build_curve(mask, args.cy, args.cx, np.deg2rad(np.arange(0, 360, 0.5)))
    P, _ = pano2.panoramic(vol, r, args.cy, args.cx, ncol=args.width)
    im = rl.to_image(np.flipud(P), args.w, args.l)
    save(im, args.out)


def cmd_render3d(args):
    vol, info = load(args.src)
    v = rl.downsample(vol, args.down)
    img = rl.volume_render(v, az=args.az, el=args.el, th=args.th, op=args.op,
                           lo=args.lo, hi=args.hi, shade=args.shade)
    im = Image.fromarray((img * 255).astype(np.uint8))
    if args.scale != 1:
        im = im.resize((im.width * args.scale, im.height * args.scale), Image.LANCZOS)
    save(im, args.out)


def cmd_spin(args):
    vol, info = load(args.src)
    v = rl.downsample(vol, args.down)
    tmp = os.path.join(os.path.dirname(os.path.abspath(args.out)), '_frames')
    os.makedirs(tmp, exist_ok=True)
    for k in range(args.frames):
        az = 360.0 * k / args.frames
        img = rl.volume_render(v, az=az, el=args.el, th=args.th, op=args.op)
        Image.fromarray((img * 255).astype(np.uint8)).save(os.path.join(tmp, f'f{k:03d}.png'))
        if k % 10 == 0:
            print(f'  {k}/{args.frames}', file=sys.stderr)
    cmd = ['ffmpeg', '-y', '-framerate', str(args.fps), '-i', os.path.join(tmp, 'f%03d.png'),
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-movflags', '+faststart', args.out]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f'  → {args.out}')


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description='牙科/医学 CT、CBCT 命令行工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument('--src', default=None, help='DICOM 文件夹（默认自动查找）')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('info', help='查看检查信息'); p.set_defaults(fn=cmd_info)

    p = sub.add_parser('slice', help='导出单张/批量切面')
    p.add_argument('--axis', default='z', choices=['z', 'y', 'x'])
    p.add_argument('--at', type=int, default=0)
    p.add_argument('--range')
    p.add_argument('--w', type=float, default=3000); p.add_argument('--l', type=float, default=800)
    p.add_argument('-o', '--out', required=True)
    p.set_defaults(fn=cmd_slice)

    p = sub.add_parser('mpr', help='导出三个方向的序列')
    p.add_argument('--step', type=int, default=1)
    p.add_argument('--w', type=float, default=3000); p.add_argument('--l', type=float, default=800)
    p.add_argument('-o', '--out', default='mpr')
    p.set_defaults(fn=cmd_mpr)

    p = sub.add_parser('montage', help='拼接多格大图')
    p.add_argument('--axis', default='z', choices=['z', 'y', 'x'])
    p.add_argument('--step', type=int, default=20)
    p.add_argument('--range')
    p.add_argument('--cols', type=int, default=6)
    p.add_argument('--tile', type=int, default=300)
    p.add_argument('--w', type=float, default=3000); p.add_argument('--l', type=float, default=800)
    p.add_argument('-o', '--out', required=True)
    p.set_defaults(fn=cmd_montage)

    p = sub.add_parser('pano', help='生成全景片')
    p.add_argument('--level', type=int, default=175, help='牙列的层号')
    p.add_argument('--cy', type=int, default=300); p.add_argument('--cx', type=int, default=277)
    p.add_argument('--width', type=int, default=1400)
    p.add_argument('--w', type=float, default=4200); p.add_argument('--l', type=float, default=1500)
    p.add_argument('-o', '--out', required=True)
    p.set_defaults(fn=cmd_pano)

    p = sub.add_parser('render3d', help='生成三维图')
    p.add_argument('--az', type=float, default=0); p.add_argument('--el', type=float, default=0)
    p.add_argument('--th', type=float, default=300)
    p.add_argument('--op', type=float, default=1.0)
    p.add_argument('--lo', type=float, default=250); p.add_argument('--hi', type=float, default=2600)
    p.add_argument('--shade', type=float, default=0.75)
    p.add_argument('--down', type=int, default=2); p.add_argument('--scale', type=int, default=1)
    p.add_argument('-o', '--out', required=True)
    p.set_defaults(fn=cmd_render3d)

    p = sub.add_parser('spin', help='生成 3D 旋转视频')
    p.add_argument('--frames', type=int, default=60); p.add_argument('--fps', type=int, default=20)
    p.add_argument('--el', type=float, default=8); p.add_argument('--th', type=float, default=300)
    p.add_argument('--op', type=float, default=1.0); p.add_argument('--down', type=int, default=2)
    p.add_argument('-o', '--out', required=True)
    p.set_defaults(fn=cmd_spin)

    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
