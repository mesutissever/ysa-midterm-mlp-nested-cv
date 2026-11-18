"""Rapor şekilleri için yardımcı çizimler."""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def _tr(n):
    """Türkçe binlik ayırıcı: 2017 -> 2.017"""
    return f"{n:,}".replace(",", ".")


def plot_mlp_architecture(model, n_num, n_cat_onehot, activation="ReLU", title=None, path=None):
    """Tam bağlantılı MLP şeması + katman bazında öğrenilebilir parametre hesabı.
    model: src.model.MLP (Linear katmanları sırayla okunur)."""
    import torch.nn as nn
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    sizes = [linears[0].in_features] + [l.out_features for l in linears]
    n_layers = len(sizes)
    max_draw = 8                                   # her katmanda çizilen en fazla nöron

    fig, ax = plt.subplots(figsize=(10, 6.2))
    gap = 5.0
    ax.set_xlim(-1.8, gap * (n_layers - 1) + 1.8); ax.set_ylim(-4.8, 5.0); ax.set_aspect("equal"); ax.axis("off")
    xs = [gap * i for i in range(n_layers)]
    colors = ["#9ecae1"] + ["#fdae6b"] * (n_layers - 2) + ["#a1d99b"]
    pos = []
    for i, (x, n) in enumerate(zip(xs, sizes)):
        k = min(n, max_draw)
        ys = [3.2 - j * (6.4 / max(k - 1, 1)) for j in range(k)] if k > 1 else [0.0]
        if n > max_draw:                            # ortadaki nöronlar üç noktayla gösterilir
            ys = ys[: k // 2] + ys[-(k // 2):]
            ax.plot([x] * 3, [-0.3, 0, 0.3], "o", ms=2.5, color="#333")
        pos.append(ys)
        for y in ys:
            ax.add_patch(Circle((x, y), 0.22, fc=colors[i], ec="#333", lw=1, zorder=3))
    for i in range(n_layers - 1):                   # bağlantılar
        for y1 in pos[i]:
            for y2 in pos[i + 1]:
                ax.plot([xs[i], xs[i + 1]], [y1, y2], color="#999", lw=0.4, alpha=0.6, zorder=1)

    labels = [f"Girdi\nd = {sizes[0]}\n({n_num} sayısal + {n_cat_onehot} one-hot)"]
    labels += [f"Gizli katman {i}\n{sizes[i]} nöron, {activation}" for i in range(1, n_layers - 1)]
    labels += ["Çıkış\n1 logit → sigmoid\nBCEWithLogitsLoss"]
    for x, lab in zip(xs, labels):
        ax.text(x, 4.3, lab, ha="center", va="center", fontsize=9.5, color="#222")

    total, lines = 0, []
    for i, l in enumerate(linears):
        p = l.in_features * l.out_features + l.out_features
        total += p
        lines.append(f"Katman {i + 1}: {l.in_features}×{l.out_features} ağırlık + {l.out_features} bias = {_tr(p)}")
        ax.text((xs[i] + xs[i + 1]) / 2, -3.75, f"{l.in_features}×{l.out_features} + {l.out_features} = {_tr(p)}",
                ha="center", va="center", fontsize=9, color="#333")
    lines.append(f"Toplam öğrenilebilir parametre = {_tr(total)}")
    ax.text(xs[-1] / 2, -4.5, f"Toplam öğrenilebilir parametre: {_tr(total)}",
            ha="center", va="center", fontsize=10.5, weight="bold", color="#222")
    if title:
        ax.set_title(title, fontsize=11)
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    return fig, lines, total
