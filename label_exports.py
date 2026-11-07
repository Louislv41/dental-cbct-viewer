"""给导出的序列图 / 三维图加上 R/L 方位标记。"""
import glob, os
from PIL import Image, ImageDraw, ImageFont

FONT = '/System/Library/Fonts/Hiragino Sans GB.ttc'
OUT = '导出图'
PAD, COLS = 6, 6
TILES = {'轴位序列.jpg': (300, 300), '冠状位序列.jpg': (300, 246), '矢状位序列.jpg': (300, 246)}


def f(sz):
    try: return ImageFont.truetype(FONT, sz)
    except Exception: return ImageFont.load_default()


def chip(d, x, y, txt, col, size=30):
    fo = f(size)
    w = d.textlength(txt, font=fo)
    d.rectangle([x - w/2 - 5, y - size*0.62, x + w/2 + 5, y + size*0.62], fill=(0, 0, 0))
    d.text((x, y), txt, font=fo, fill=col, anchor='mm')


def label_montage(path, tile, kind):
    im = Image.open(path).convert('RGB')
    d = ImageDraw.Draw(im)
    cols = COLS
    tw, th = tile
    rows = (im.height - PAD) // (th + PAD)
    for r in range(rows):
        for c in range(cols):
            x0 = PAD + c*(tw + PAD); y0 = PAD + r*(th + PAD)
            if x0 + tw > im.width or y0 + th > im.height: continue
            ym = y0 + th/2
            if kind == 'sag':
                chip(d, x0 + 34, ym, '前', (255, 206, 77))
                chip(d, x0 + tw - 34, ym, '后', (255, 138, 138))
            else:
                chip(d, x0 + 30, ym, 'R', (255, 206, 77))
                chip(d, x0 + tw - 30, ym, 'L', (255, 138, 138))
    im.save(path, quality=92, subsampling=0)
    print('  labeled', path)


def label_3d(path, az):
    im = Image.open(path).convert('RGB')
    d = ImageDraw.Draw(im)
    flip = not (-90 < az < 90)
    chip(d, 40, im.height/2, 'L' if flip else 'R', (255, 206, 77), 34)
    chip(d, im.width - 40, im.height/2, 'R' if flip else 'L', (255, 138, 138), 34)
    im.save(path, quality=92, subsampling=0)
    print('  labeled', path)


for name, tile in TILES.items():
    p = os.path.join(OUT, name)
    if os.path.exists(p):
        label_montage(p, tile, 'sag' if '矢状' in name else 'lr')

for name, az in [('三维_正面.jpg', 0), ('三维_左前45°.jpg', 35), ('三维_右前45°.jpg', -35),
                 ('三维_左侧.jpg', 90), ('三维_后方.jpg', 180)]:
    p = os.path.join(OUT, name)
    if os.path.exists(p):
        label_3d(p, az)
