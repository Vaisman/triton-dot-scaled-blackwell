import sys
# Data: results/vm-b300-20261116 SUMMARY (8192^3, TFLOP/s, median of 3)
rows = [("128×128×128, 4w, 3s", 185.6, 1075.3),
        ("128×128×128, 8w, 3s", 185.0, 1027.7),
        ("128×128×64, 4w, 4s", 363.5, 723.2),
        ("128×256×128, 8w, 3s", 209.4, 1213.4)]
themes = {
 "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
               fb="#eb6834", nat="#2a78d6"),
 "dark":  dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
               fb="#d95926", nat="#3987e5"),
}
W, LEFT, RIGHT = 760, 170, 70
TOP, BAR, GAP_IN, GAP_GRP = 86, 18, 2, 22
XMAX = 1400
plotw = W - LEFT - RIGHT
def x(v): return LEFT + v / XMAX * plotw
def svg(t):
    o = []
    H = TOP + len(rows) * (2 * BAR + GAP_IN + GAP_GRP) + 62
    a = o.append
    a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif" role="img" aria-labelledby="t d">')
    a(f'<title id="t">tl.dot_scaled on B300: missing scale vs unit scale</title>')
    a(f'<desc id="d">Throughput in TFLOP/s at M=N=K=8192 for four tile configs. Missing scale (bf16 fallback, kind::f16): 186, 185, 364, 209. Unit scale (native kind::mxf8f6f4): 1075, 1028, 723, 1213.</desc>')
    a(f'<rect width="{W}" height="{H}" fill="{t["surface"]}"/>')
    a(f'<text x="{LEFT}" y="28" font-size="16" font-weight="600" fill="{t["ink"]}">tl.dot_scaled on B300, M = N = K = 8192 (TFLOP/s, higher is better)</text>')
    # legend
    lx = LEFT
    for name, col in (("missing scale → bf16 fallback (kind::f16)", t["fb"]), ("unit scale → native (kind::mxf8f6f4)", t["nat"])):
        a(f'<rect x="{lx}" y="44" width="12" height="12" rx="3" fill="{col}"/>')
        a(f'<text x="{lx+18}" y="54" font-size="12.5" fill="{t["ink2"]}">{name}</text>')
        lx += 18 + len(name) * 6.9 + 24
    # grid
    bottom = H - 56
    for v in range(0, XMAX + 1, 200):
        a(f'<line x1="{x(v):.1f}" y1="{TOP-8}" x2="{x(v):.1f}" y2="{bottom}" stroke="{t["grid"] if v else t["axis"]}" stroke-width="1"/>')
        a(f'<text x="{x(v):.1f}" y="{bottom+16}" font-size="11" fill="{t["muted"]}" text-anchor="middle">{v}</text>')
    a(f'<text x="{W-8}" y="{TOP-12}" font-size="11.5" fill="{t["muted"]}" text-anchor="end">speedup</text>')
    y = TOP
    for label, fb, nat in rows:
        gy = y + BAR + GAP_IN / 2
        a(f'<text x="{LEFT-10}" y="{gy+4}" font-size="12.5" fill="{t["ink2"]}" text-anchor="end">{label}</text>')
        for v, col, yy in ((fb, t["fb"], y), (nat, t["nat"], y + BAR + GAP_IN)):
            w = x(v) - LEFT
            # 4px rounded data end, square at the baseline
            a(f'<path d="M{LEFT},{yy} h{w-4:.1f} a4,4 0 0 1 4,4 v{BAR-8} a4,4 0 0 1 -4,4 h{-(w-4):.1f} z" fill="{col}"/>')
            a(f'<text x="{x(v)+6:.1f}" y="{yy+BAR-5}" font-size="11.5" fill="{t["ink"]}">{v:.0f}</text>')
        a(f'<text x="{W-8}" y="{gy+4}" font-size="12" font-weight="600" fill="{t["ink2"]}" text-anchor="end">{nat/fb:.1f}×</text>')
        y += 2 * BAR + GAP_IN + GAP_GRP
    a(f'<text x="12" y="{H-12}" font-size="11" fill="{t["muted"]}">e4m3 × e4m3; LHS scale per 32-element block; RHS scale None vs tl.full(…, 127). Median of 3 runs.</text>')
    a('</svg>')
    return "\n".join(o)
for k, t in themes.items():
    open(f"figures/throughput-{k}.svg", "w").write(svg(t))
print("ok")
