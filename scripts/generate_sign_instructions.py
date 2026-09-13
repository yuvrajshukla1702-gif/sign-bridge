"""
ASL finger-spelling teaching copy + regenerate dictionary cards
with formation instructions (not just the letter glyph).
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "references" / "sign_instructions.json"
DICT = ROOT / "assets" / "signs_dict"

# Plain-language how-to for beginners (ASL finger-spelling style)
HOW_TO = {
    "A": {
        "title": "Letter A",
        "steps": [
            "Make a fist (curl all four fingers into your palm).",
            "Rest your thumb against the side of your index finger — not tucked inside the fist.",
            "Face your palm outward (toward the camera / other person).",
        ],
        "remember": "Fist + thumb on the side = A",
    },
    "B": {
        "title": "Letter B",
        "steps": [
            "Hold all four fingers straight up and together (flat hand).",
            "Bend your thumb across your palm.",
            "Palm faces outward.",
        ],
        "remember": "Flat hand, thumb tucked = B",
    },
    "C": {
        "title": "Letter C",
        "steps": [
            "Curve your whole hand into a C shape.",
            "Thumb and fingers form an open curve (like holding a cup).",
            "Palm faces sideways.",
        ],
        "remember": "Curved hand like the letter C",
    },
    "D": {
        "title": "Letter D",
        "steps": [
            "Point your index finger straight up.",
            "Touch your thumb to the tips of the other three fingers (they form a circle/O).",
            "Palm faces outward.",
        ],
        "remember": "Index up + other fingers touch thumb = D",
    },
    "E": {
        "title": "Letter E",
        "steps": [
            "Curl all four fingertips down toward your palm.",
            "Tuck the thumb against the fingertips (not over the top like S).",
            "Palm faces outward.",
        ],
        "remember": "Fingertips on palm edge, thumb in = E",
    },
    "F": {
        "title": "Letter F",
        "steps": [
            "Touch tip of thumb to tip of index finger (OK shape).",
            "Keep middle, ring, and pinky fingers straight up.",
            "Palm faces outward.",
        ],
        "remember": "OK circle + three fingers up = F",
    },
    "G": {
        "title": "Letter G",
        "steps": [
            "Point index finger sideways (horizontal).",
            "Extend thumb roughly parallel under it.",
            "Curl the other fingers into the palm. Palm faces you / sideways.",
        ],
        "remember": "Index + thumb pointing sideways = G",
    },
    "H": {
        "title": "Letter H",
        "steps": [
            "Extend index and middle fingers together, pointing sideways.",
            "Thumb holds down the other fingers.",
            "Palm faces you.",
        ],
        "remember": "Two fingers together, sideways = H",
    },
    "I": {
        "title": "Letter I",
        "steps": [
            "Make a fist.",
            "Stick only your pinky finger straight up.",
            "Palm faces outward.",
        ],
        "remember": "Pinky up = I",
    },
    "J": {
        "title": "Letter J",
        "steps": [
            "Start like letter I (pinky up).",
            "Draw a J in the air with your pinky (sweep down and hook).",
            "Static practice: hold the I handshape.",
        ],
        "remember": "Pinky draws a J = J",
    },
    "K": {
        "title": "Letter K",
        "steps": [
            "Index finger points up.",
            "Middle finger points up/out at an angle (like a V but thumb touches between them).",
            "Thumb tip rests between index and middle at the base. Palm out.",
        ],
        "remember": "K looks like a V with thumb in the middle",
    },
    "L": {
        "title": "Letter L",
        "steps": [
            "Extend index finger straight up.",
            "Extend thumb straight out to the side.",
            "Curl other fingers down. Palm faces outward.",
        ],
        "remember": "Index + thumb make an L shape",
    },
    "M": {
        "title": "Letter M",
        "steps": [
            "Tuck thumb under three fingers (index, middle, ring).",
            "Pinky curls in.",
            "Palm faces down/out — three knuckles show over the thumb.",
        ],
        "remember": "Thumb under 3 fingers = M",
    },
    "N": {
        "title": "Letter N",
        "steps": [
            "Tuck thumb under two fingers (index and middle).",
            "Ring and pinky curl in.",
            "Palm faces down/out.",
        ],
        "remember": "Thumb under 2 fingers = N",
    },
    "O": {
        "title": "Letter O",
        "steps": [
            "Curve all fingertips to meet the thumb, making a round O.",
            "Like holding a small ball.",
            "Palm faces sideways.",
        ],
        "remember": "Fingers + thumb make an O circle",
    },
    "P": {
        "title": "Letter P",
        "steps": [
            "Form a K handshape.",
            "Point it downward (middle finger aims down).",
            "Palm faces down.",
        ],
        "remember": "K pointing down = P",
    },
    "Q": {
        "title": "Letter Q",
        "steps": [
            "Form a G handshape.",
            "Point index and thumb downward.",
            "Palm faces down.",
        ],
        "remember": "G pointing down = Q",
    },
    "R": {
        "title": "Letter R",
        "steps": [
            "Cross index and middle fingers (like wishing for luck).",
            "Thumb holds the other fingers.",
            "Palm faces outward.",
        ],
        "remember": "Crossed index + middle = R",
    },
    "S": {
        "title": "Letter S",
        "steps": [
            "Make a fist.",
            "Wrap the thumb across the front of the fingers (over the knuckles).",
            "Palm faces outward.",
        ],
        "remember": "Fist with thumb over fingers = S (different from A)",
    },
    "T": {
        "title": "Letter T",
        "steps": [
            "Make a fist.",
            "Tuck the thumb between index and middle finger.",
            "Palm faces outward.",
        ],
        "remember": "Thumb between index & middle = T",
    },
    "U": {
        "title": "Letter U",
        "steps": [
            "Index and middle fingers straight up, pressed together.",
            "Thumb holds ring and pinky down.",
            "Palm faces outward.",
        ],
        "remember": "Two fingers up, together = U",
    },
    "V": {
        "title": "Letter V",
        "steps": [
            "Index and middle fingers up in a V (peace sign).",
            "Thumb holds other fingers down.",
            "Palm faces outward.",
        ],
        "remember": "Peace / victory V = V",
    },
    "W": {
        "title": "Letter W",
        "steps": [
            "Index, middle, and ring fingers up and spread.",
            "Thumb holds pinky (or touches pinky).",
            "Palm faces outward.",
        ],
        "remember": "Three fingers up = W",
    },
    "X": {
        "title": "Letter X",
        "steps": [
            "Make a fist with index finger raised.",
            "Bend/hook the index finger (like a claw).",
            "Palm faces outward.",
        ],
        "remember": "Hooked index finger = X",
    },
    "Y": {
        "title": "Letter Y",
        "steps": [
            "Extend thumb and pinky out (hang-loose / shaka shape).",
            "Curl middle three fingers into the palm.",
            "Palm faces outward.",
        ],
        "remember": "Thumb + pinky out = Y",
    },
    "Z": {
        "title": "Letter Z",
        "steps": [
            "Point with index finger (like D but used for drawing).",
            "Draw a Z in the air: across, diagonal, across.",
            "Static practice: hold index finger ready to write Z.",
        ],
        "remember": "Index finger draws Z = Z",
    },
    "0": {
        "title": "Number 0",
        "steps": ["Same handshape as letter O — fingertips meet thumb in a circle."],
        "remember": "O shape = 0",
    },
    "1": {
        "title": "Number 1",
        "steps": ["Index finger up; other fingers curled (like D without the O circle emphasis)."],
        "remember": "One finger up = 1",
    },
    "2": {
        "title": "Number 2",
        "steps": ["Index and middle up in a V (same idea as letter V)."],
        "remember": "Two fingers = 2",
    },
    "3": {
        "title": "Number 3",
        "steps": ["Thumb, index, and middle extended; ring and pinky down."],
        "remember": "Thumb + 2 fingers = 3",
    },
    "4": {
        "title": "Number 4",
        "steps": ["Four fingers up, thumb tucked across palm."],
        "remember": "Four fingers = 4",
    },
    "5": {
        "title": "Number 5",
        "steps": ["Open hand — all five digits spread, palm out."],
        "remember": "Open hand = 5",
    },
    "6": {
        "title": "Number 6",
        "steps": ["Touch thumb tip to pinky tip; other three fingers up."],
        "remember": "Thumb touches pinky = 6",
    },
    "7": {
        "title": "Number 7",
        "steps": ["Touch thumb tip to ring fingertip; other fingers up."],
        "remember": "Thumb touches ring = 7",
    },
    "8": {
        "title": "Number 8",
        "steps": ["Touch thumb tip to middle fingertip; other fingers up."],
        "remember": "Thumb touches middle = 8",
    },
    "9": {
        "title": "Number 9",
        "steps": ["Touch thumb tip to index fingertip (like F); other fingers up."],
        "remember": "Thumb touches index = 9",
    },
}


def draw_card(label: str, info: dict, path: Path) -> None:
    img = Image.new("RGB", (420, 520), "#0b1c2c")
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([14, 14, 406, 506], radius=28, outline="#1aa6a0", width=3)
    d.text((210, 48), info["title"], fill="#5ed1cb", anchor="mm")
    d.text((210, 88), f"How to sign “{label}”", fill="#8fa3b5", anchor="mm")

    # Simple instructional icons by type
    y = 130
    for i, step in enumerate(info["steps"][:4], start=1):
        d.ellipse([36, y, 58, y + 22], outline="#1aa6a0", width=2)
        d.text((47, y + 11), str(i), fill="#5ed1cb", anchor="mm")
        # wrap roughly
        words = step.split()
        line = ""
        ly = y
        for w in words:
            test = (line + " " + w).strip()
            if len(test) > 38:
                d.text((70, ly), line, fill="#e8eef2")
                ly += 18
                line = w
            else:
                line = test
        if line:
            d.text((70, ly), line, fill="#e8eef2")
        y = max(y + 70, ly + 28)

    d.rounded_rectangle([28, 430, 392, 480], radius=14, fill="#122a40")
    d.text((210, 455), info["remember"], fill="#f0c45c", anchor="mm")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(HOW_TO, indent=2), encoding="utf-8")

    for label, info in HOW_TO.items():
        folder = "alphabet" if label.isalpha() else "numbers"
        draw_card(label, info, DICT / folder / f"{label}.png")

    print(f"Wrote {OUT_JSON}")
    print(f"Updated cards in {DICT}")


if __name__ == "__main__":
    main()
