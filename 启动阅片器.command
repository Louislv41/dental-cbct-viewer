#!/bin/bash
# 双击本文件即可启动 CT 阅片器（会自动打开浏览器）
cd "$(dirname "$0")" || exit 1

PY=$(command -v python3)
if [ -z "$PY" ]; then
  echo "没找到 python3，请先安装 Python 3（https://www.python.org）"
  read -r -p "按回车退出…"; exit 1
fi

# 自动寻找同目录或上一级目录里的 DICOM 文件夹
DATA="${1:-}"
if [ -z "$DATA" ]; then
  for d in ../*/ ; do
    if ls "$d"*.dcm >/dev/null 2>&1; then DATA="$d"; break; fi
  done
fi
if [ -z "$DATA" ]; then
  for d in */ ; do
    if ls "$d"*.dcm >/dev/null 2>&1; then DATA="$d"; break; fi
  done
fi

echo "══════════════════════════════════════════"
echo "  CT / CBCT 阅片器"
echo "  数据目录: ${DATA:-未找到 .dcm 文件}"
echo "══════════════════════════════════════════"

"$PY" -c "import numpy, scipy, PIL" 2>/dev/null
if [ $? -ne 0 ]; then
  echo "缺少依赖，正在安装（只需要一次）…"
  "$PY" -m pip install --quiet numpy scipy pillow || {
    echo "安装失败。请手动运行: python3 -m pip install numpy scipy pillow"; read -r -p "按回车退出…"; exit 1; }
fi

"$PY" app.py "$DATA" 8788
