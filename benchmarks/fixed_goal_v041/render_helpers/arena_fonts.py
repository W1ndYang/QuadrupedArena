"""Presentation font lookup; CPU rules/tests must not require OS font packages.

Install Noto CJK for readable Chinese labels. A fallback may display missing
glyph boxes, but should not prevent a headless experiment or test from running.
"""
import os
from pathlib import Path
from PIL import ImageFont


def font_candidates(bold=False):
    paths = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    if bold:
        paths.insert(0, '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
    return paths


def load_font(size, bold=False):
    override = os.environ.get('ARENA_FONT')
    if override:
        return ImageFont.truetype(str(Path(override).expanduser()), size)
    for candidate in font_candidates(bold):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)
