"""
Generate ASL visual cue cards — filled hand drawings from landmark templates.

Outputs:
  - assets/signs_dict/alphabet/A-Z.png
  - assets/signs_dict/numbers/0-9.png
  - assets/signs/*.png  and  assets/signs_dict/phrases/*.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "references" / "static_landmarks.json"
INSTR = ROOT / "data" / "references" / "sign_instructions.json"
SENT = ROOT / "data" / "references" / "sentences.json"

ALPHA_DIR = ROOT / "assets" / "signs_dict" / "alphabet"
NUM_DIR = ROOT / "assets" / "signs_dict" / "numbers"
PHRASE_DICT = ROOT / "assets" / "signs_dict" / "phrases"
SIGNS_DIR = ROOT / "assets" / "signs"

BG = (248, 250, 252)
PANEL = (226, 232, 240)
INK = (15, 23, 42)
MUTED = (71, 85, 105)
ACCENT = (14, 116, 144)
ACCENT_SOFT = (8, 145, 178)
SKIN = (232, 196, 164)
SKIN_D = (196, 150, 112)
SKIN_LINE = (120, 80, 50)
NAIL = (245, 220, 200)

# MediaPipe hand topology
FINGERS = [
    [0, 1, 2, 3, 4],       # thumb
    [0, 5, 6, 7, 8],       # index
    [0, 9, 10, 11, 12],    # middle
    [0, 13, 14, 15, 16],   # ring
    [0, 17, 18, 19, 20],   # pinky
]
PALM_RING = [0, 1, 5, 9, 13, 17]
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

PHRASE_TITLES = {
    "hello": ("HELLO", "Open flat hand — wave from temple"),
    "thank_you": ("THANK YOU", "Flat hand at chin → move forward"),
    "yes": ("YES", "Fist nods like a head nodding"),
    "no": ("NO", "Index + middle snap toward thumb"),
    "water": ("WATER", "W-hand taps the chin"),
    "need_help": ("NEED HELP", "Open palm up — offer help"),
    "hospital": ("HOSPITAL", "H-hand draws + on upper arm"),
    "pain": ("PAIN", "Index tip points / jabs to pain"),
    "emergency": ("EMERGENCY", "E-handshape — urgent attention"),
    "medical_help": ("MEDICAL HELP", "M-hand — medical / help cue"),
    "family": ("FAMILY", "F-handshape — family circle"),
    "iloveyou": ("I LOVE YOU", "Thumb + index + pinky (ILY)"),
}


def font(size: int) -> ImageFont.ImageFont:
    for name in (
        r"C:\Windows\Fonts\segoeuib.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        p = Path(name)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def load_landmarks() -> Dict[str, List[float]]:
    if not REF.exists():
        return {}
    data = json.loads(REF.read_text(encoding="utf-8"))
    return data.get("landmarks", {})


def pts_from_flat(flat: List[float]) -> List[Tuple[float, float]]:
    return [(float(flat[i * 3]), float(flat[i * 3 + 1])) for i in range(21)]


def map_pts(
    pts: List[Tuple[float, float]], cx: int, cy: int, scale: float
) -> List[Tuple[int, int]]:
    out = []
    for x, y in pts:
        out.append((int(cx + x * scale), int(cy + y * scale)))
    return out


def _thick_segment(
    draw: ImageDraw.ImageDraw,
    a: Tuple[int, int],
    b: Tuple[int, int],
    width: float,
    fill,
) -> None:
    """Filled capsule between two points (no contrasting joint dots)."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length, dx / length
    hw = width / 2
    poly = [
        (ax + nx * hw, ay + ny * hw),
        (ax - nx * hw, ay - ny * hw),
        (bx - nx * hw, by - ny * hw),
        (bx + nx * hw, by + ny * hw),
    ]
    draw.polygon(poly, fill=fill)
    r = max(2, int(hw))
    draw.ellipse([ax - r, ay - r, ax + r, ay + r], fill=fill)
    draw.ellipse([bx - r, by - r, bx + r, by + r], fill=fill)


def draw_filled_hand(
    draw: ImageDraw.ImageDraw,
    pts: List[Tuple[float, float]],
    cx: int,
    cy: int,
    scale: float = 210.0,
) -> None:
    """Teaching-card hand: soft filled fingers + palm (readable pose, not skeleton)."""
    m = map_pts(pts, cx, cy, scale)

    # Shadow
    for fi, chain in enumerate(FINGERS):
        for si in range(len(chain) - 1):
            a = (m[chain[si]][0] + 6, m[chain[si]][1] + 10)
            b = (m[chain[si + 1]][0] + 6, m[chain[si + 1]][1] + 10)
            _thick_segment(draw, a, b, 28 if fi == 0 else 22, (205, 214, 224))
    palm_s = [(m[i][0] + 6, m[i][1] + 10) for i in (0, 1, 5, 9, 13, 17)]
    draw.polygon(palm_s, fill=(205, 214, 224))

    # Palm
    palm = [m[i] for i in (0, 1, 5, 9, 13, 17)]
    draw.polygon(palm, fill=SKIN)
    wx, wy = m[0]
    draw.ellipse([wx - 38, wy - 4, wx + 38, wy + 54], fill=SKIN)
    hx = sum(p[0] for p in palm) // len(palm)
    hy = sum(p[1] for p in palm) // len(palm) - 6
    draw.ellipse([hx - 30, hy - 22, hx + 30, hy + 28], fill=SKIN_D)

    # Fingers — wide overlapping fills only
    widths = {
        0: (34, 30, 26, 22),
        1: (26, 23, 20, 17),
        2: (28, 24, 21, 18),
        3: (26, 23, 20, 17),
        4: (22, 19, 16, 14),
    }
    for fi, chain in enumerate(FINGERS):
        segs = widths[fi]
        for si in range(len(chain) - 1):
            a, b = m[chain[si]], m[chain[si + 1]]
            _thick_segment(draw, a, b, segs[min(si, len(segs) - 1)], SKIN)

    # Soft creases only (no bone lines)
    for idx in (6, 10, 14, 18):
        x, y = m[idx]
        draw.arc([x - 9, y - 5, x + 9, y + 5], 200, 340, fill=SKIN_LINE, width=2)

    # Nails
    for tip in (4, 8, 12, 16, 20):
        x, y = m[tip]
        draw.ellipse([x - 8, y - 10, x + 8, y + 4], fill=NAIL, outline=SKIN_D)

def card_base(w: int, h: int, title: str, subtitle: str) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)
    # soft radial atmosphere
    for i in range(6):
        r = 60 + i * 36
        draw.ellipse(
            [w // 2 - r, 120 - r // 2, w // 2 + r, 120 + r],
            outline=(226, 232, 240),
        )
    draw.rounded_rectangle([14, 14, w - 14, h - 14], radius=24, outline=ACCENT, width=3)
    draw.text((32, 28), title, font=font(48), fill=INK)
    draw.text((34, 88), subtitle[:52], font=font(18), fill=MUTED)
    return img, draw


def make_static_card(key: str, flat: List[float], remember: str, out: Path) -> None:
    w, h = 480, 600
    title = key
    img, draw = card_base(w, h, title, remember or f"Hold palm toward camera — sign {key}")
    pts = pts_from_flat(flat)
    draw_filled_hand(draw, pts, cx=w // 2, cy=330, scale=235)
    draw.rounded_rectangle([36, h - 100, w - 36, h - 36], radius=14, fill=PANEL)
    tip = remember or "Match this handshape, then hold still"
    draw.text((50, h - 78), tip[:44], font=font(17), fill=MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_png(img, out)


def make_phrase_card(key: str, flat: List[float], title: str, tip: str, out: Path) -> None:
    w, h = 520, 580
    img, draw = card_base(w, h, title, tip)
    pts = pts_from_flat(flat)
    draw_filled_hand(draw, pts, cx=w // 2, cy=310, scale=220)
    draw.rounded_rectangle([36, h - 100, w - 36, h - 36], radius=14, fill=PANEL)
    draw.text((50, h - 78), tip[:48], font=font(17), fill=MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_png(img, out)


def _save_png(img: Image.Image, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".png.tmp")
    img.save(tmp, "PNG")
    try:
        tmp.replace(out)
    except OSError:
        alt = out.with_name(out.stem + "_new.png")
        try:
            tmp.replace(alt)
            print(f"WARN: locked {out.name}; wrote {alt.name}")
        except OSError:
            print(f"WARN: failed to write {out}")
            if tmp.exists():
                tmp.unlink(missing_ok=True)


def main() -> None:
    landmarks = load_landmarks()
    instructions = {}
    if INSTR.exists():
        instructions = json.loads(INSTR.read_text(encoding="utf-8"))

    for key, flat in landmarks.items():
        remember = instructions.get(key, {}).get("remember", "")
        if key.isalpha() and len(key) == 1:
            make_static_card(key, flat, remember, ALPHA_DIR / f"{key}.png")
        elif key.isdigit():
            make_static_card(key, flat, remember, NUM_DIR / f"{key}.png")

    for key, (title, tip) in PHRASE_TITLES.items():
        flat = landmarks.get(key)
        if not flat:
            # fallback: letter-like if phrase missing
            continue
        remember = instructions.get(key, {}).get("remember", tip)
        make_phrase_card(key, flat, title, remember or tip, SIGNS_DIR / f"{key}.png")
        make_phrase_card(key, flat, title, remember or tip, PHRASE_DICT / f"{key}.png")

    if SENT.exists():
        sent = json.loads(SENT.read_text(encoding="utf-8"))
        phrases = sent.setdefault("phrases", {})
        for key, (title, tip) in PHRASE_TITLES.items():
            phrases[key] = {
                "title": title,
                "image": f"/assets/signs/{key}.png",
                "remember": tip,
                "steps": [
                    "Show your hand clearly to the camera.",
                    f"Form the “{title}” handshape as in the cue card.",
                    "Hold steady until the match bar turns green.",
                ],
            }
        SENT.write_text(json.dumps(sent, indent=2), encoding="utf-8")

    print(f"Cue cards written under {ALPHA_DIR}, {NUM_DIR}, {SIGNS_DIR}, {PHRASE_DICT}")


if __name__ == "__main__":
    main()
