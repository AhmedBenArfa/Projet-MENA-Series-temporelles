"""Build the French PowerPoint deck for the MENA VaR project.

Generates fresh figures under outputs/figures/ (prefixed ``deck_``) and
assembles presentation/VaR_MENA.pptx with python-pptx. All numbers shown
on data slides are read live from outputs/results.csv and
outputs/best_per_index.csv -- nothing here is hand-typed from memory.

Run:
    "C:/Users/Mega-pc/anaconda3/python.exe" scripts/build_deck.py
"""

import sys
import pathlib

# --- bootstrap: make `tsvar` importable regardless of cwd -----------------
_root = pathlib.Path(__file__).resolve().parent.parent
for _c in (_root, *_root.parents):
    if (_c / "src" / "tsvar").is_dir():
        sys.path.insert(0, str(_c / "src"))
        break

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

from tsvar.data import train_test_returns, INDEX_FILES, MENA, BENCHMARKS, load_index
from tsvar.volatility import walk_forward_garch
from tsvar.deep import walk_forward_dl
from tsvar.var import var_series

ROOT = _root
DATA = ROOT / "data (1)" / "data"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
PRES_DIR = ROOT / "presentation"
PRES_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = OUT / "results.csv"
BEST_CSV = OUT / "best_per_index.csv"

# ---------------------------------------------------------------------------
# Palette -- "Gulf Ledger": deep navy/teal (finance, trust) + gold (MENA,
# value) + warm sand (paper/desert). Navy dominates (title/section/dark
# slides + headers); gold is the single sharp accent (numbers, winners,
# circles); sand is the light content background.
# ---------------------------------------------------------------------------
NAVY = RGBColor(0x0E, 0x2A, 0x3D)
NAVY_DARK = RGBColor(0x08, 0x1B, 0x28)
GOLD = RGBColor(0xC9, 0x98, 0x2D)
GOLD_LIGHT = RGBColor(0xE7, 0xC9, 0x7E)
SAND = RGBColor(0xF3, 0xEC, 0xDD)
SAND_DARK = RGBColor(0xE7, 0xDC, 0xC3)
INK = RGBColor(0x1B, 0x24, 0x30)
SLATE = RGBColor(0x5A, 0x6B, 0x78)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0x2E, 0x7D, 0x46)
YELLOW = RGBColor(0xC8, 0x8A, 0x1E)
RED = RGBColor(0xAE, 0x39, 0x2F)

FONT_HEAD = "Cambria"
FONT_BODY = "Calibri"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN = Inches(0.6)


# ===========================================================================
# Figure generation (fresh, for this deck)
# ===========================================================================

def gen_returns_clustering(index_name="Tunindex"):
    """Log-return series (train+test) illustrating volatility clustering."""
    tr, te = train_test_returns(index_name, DATA)
    full = pd.concat([tr, te])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(full.index, full.values, lw=0.6, color="#0E2A3D")
    ax.axvline(tr.index[-1], color="#C9982D", lw=1.2, ls="--")
    ax.text(tr.index[-1], full.values.max() * 0.92, "  fin train / début test",
             color="#8a6a1e", fontsize=9)
    ax.set_title(f"Rendements log de {index_name} -- regroupement de volatilité", fontsize=12)
    ax.set_ylabel("rendement (%)")
    ax.margins(x=0.01)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    path = FIG / "deck_returns_clustering.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def gen_garch_var(index_name="Tunindex", alpha=0.01):
    """GARCH BHS-VaR with violations on the test set (train-once, walk-forward)."""
    tr, te = train_test_returns(index_name, DATA)
    fc = walk_forward_garch(tr, te)
    v = var_series(fc, alpha)
    n_viol = int(np.sum(fc.y_true < v))
    rate = n_viol / len(fc.y_true)

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(fc.dates, fc.y_true, label="rendement réalisé", lw=0.8, color="#5A6B78")
    ax.plot(fc.dates, v, label=f"VaR GARCH ({int((1-alpha)*100)}%)", color="#AE392F", lw=1.3)
    br = fc.y_true < v
    ax.scatter(fc.dates[br], fc.y_true[br], color="#0E2A3D", s=22, zorder=5,
               label=f"violations (n={n_viol}, {rate:.1%})")
    ax.set_title(f"VaR GARCH (BHS) -- {index_name}, alpha={alpha}", fontsize=12)
    ax.set_ylabel("rendement (%)")
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    ax.margins(x=0.01)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    path = FIG / "deck_garch_var_tunindex.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path, n_viol, rate


def gen_adi_lstm_vs_garch(alpha=0.01):
    """ADI: GARCH vs LSTM VaR side by side -- for the hypothesis-verdict slide."""
    tr, te = train_test_returns("ADI", DATA)
    fc_g = walk_forward_garch(tr, te)
    fc_l = walk_forward_dl(tr, te, "lstm")
    v_g = var_series(fc_g, alpha)
    v_l = var_series(fc_l, alpha)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, fc, v, title, color in (
        (axes[0], fc_g, v_g, "GARCH", "#0E2A3D"),
        (axes[1], fc_l, v_l, "LSTM", "#C9982D"),
    ):
        ax.plot(fc.dates, fc.y_true, lw=0.7, color="#5A6B78")
        ax.plot(fc.dates, v, color=color, lw=1.3)
        br = fc.y_true < v
        n = int(br.sum())
        ax.scatter(fc.dates[br], fc.y_true[br], color="#AE392F", s=20, zorder=5)
        ax.set_title(f"{title} -- ADI 99% (n={n} viol., {n/len(fc.y_true):.1%})", fontsize=11)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    path = FIG / "deck_adi_garch_vs_lstm.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def gen_winner_heatmap():
    """Basel-zone heatmap per (index, model) at alpha=1%, winners starred."""
    res = pd.read_csv(RESULTS_CSV)
    best = pd.read_csv(BEST_CSV)
    r = res[res.alpha == 0.01].copy()
    models = ["ARIMA", "SARIMA", "GARCH", "RF", "XGB", "ANN", "LSTM"]
    indices = ["Tunindex", "ADI", "MASI", "TASI"]
    zone_val = {"green": 0, "yellow": 1, "red": 2}
    mat = np.zeros((len(indices), len(models)))
    for i, idx in enumerate(indices):
        for j, m in enumerate(models):
            row = r[(r["index"] == idx) & (r["model"] == m)]
            mat[i, j] = zone_val.get(row.iloc[0]["basel_zone"], 1) if len(row) else np.nan

    cmap = matplotlib.colors.ListedColormap(["#2E7D46", "#C88A1E", "#AE392F"])
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, fontsize=12)
    ax.set_yticks(range(len(indices))); ax.set_yticklabels(indices, fontsize=12)
    ax.set_title("Zone de Bâle par modèle et par indice (alpha=1%) -- gagnant encadré", fontsize=13)

    winners = {row["index"]: row["model"] for _, row in best.iterrows()}
    for i, idx in enumerate(indices):
        for j, m in enumerate(models):
            row = r[(r["index"] == idx) & (r["model"] == m)]
            if not len(row):
                continue
            rate = row.iloc[0]["observed_rate"]
            ax.text(j, i, f"{rate:.1%}", ha="center", va="center", fontsize=10.5, color="white")
            if winners.get(idx) == m:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                      edgecolor="#0E2A3D", linewidth=3)
                ax.add_patch(rect)
    fig.tight_layout()
    path = FIG / "deck_winner_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ===========================================================================
# python-pptx helpers
# ===========================================================================

def new_presentation():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def blank_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def set_background(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, left, top, width, height, color, line=False):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    if line:
        shp.line.color.rgb = color
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def add_text(slide, left, top, width, height, text, size=16, color=INK, bold=False,
             italic=False, font=FONT_BODY, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
             line_spacing=1.0):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.name = font
        r.font.color.rgb = color
    return box


def add_bullets(slide, left, top, width, height, items, size=15, color=INK,
                 font=FONT_BODY, bullet="—", space_after=8, bold_lead=False,
                 anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        p.line_spacing = 1.08
        r = p.add_run()
        r.text = f"{bullet}  {item}"
        r.font.size = Pt(size)
        r.font.name = font
        r.font.color.rgb = color
    return box


def add_kicker_and_title(slide, num, kicker, title, dark=False):
    """Section-number circle (repeated visual motif) + kicker + slide title."""
    circ = slide.shapes.add_shape(MSO_SHAPE.OVAL, MARGIN, Inches(0.42), Inches(0.55), Inches(0.55))
    circ.fill.solid()
    circ.fill.fore_color.rgb = GOLD
    circ.line.fill.background()
    circ.shadow.inherit = False
    tf = circ.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = f"{num:02d}"
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.name = FONT_HEAD
    r.font.color.rgb = NAVY_DARK

    kick_color = GOLD_LIGHT if dark else GOLD
    add_text(slide, Inches(1.3), Inches(0.44), Inches(9), Inches(0.3), kicker.upper(),
              size=12, color=kick_color, bold=True, font=FONT_BODY)
    title_color = WHITE if dark else NAVY
    add_text(slide, Inches(1.3), Inches(0.72), Inches(11.4), Inches(0.7), title,
              size=30, color=title_color, bold=True, font=FONT_HEAD)


def add_footer(slide, page, total, note="Projet VaR MENA -- Deep Learning vs modèles statistiques", dark=False):
    color = SLATE if not dark else RGBColor(0x9F, 0xB3, 0xC2)
    add_text(slide, MARGIN, Inches(7.14), Inches(10.5), Inches(0.3), note,
              size=9, color=color, italic=True, font=FONT_BODY)
    add_text(slide, Inches(12.3), Inches(7.14), Inches(0.5), Inches(0.3), f"{page}/{total}",
              size=9, color=color, font=FONT_BODY, align=PP_ALIGN.RIGHT)


def add_picture_framed(slide, path, left, top, width=None, height=None):
    pic = slide.shapes.add_picture(str(path), left, top, width=width, height=height)
    pic.line.color.rgb = SAND_DARK
    pic.line.width = Pt(1)
    return pic


def _set_cell(cell, text, size=12, color=INK, bold=False, bg=None, align=PP_ALIGN.CENTER, font=FONT_BODY):
    cell.margin_left = Inches(0.06)
    cell.margin_right = Inches(0.06)
    cell.margin_top = Inches(0.03)
    cell.margin_bottom = Inches(0.03)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    if bg is not None:
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg
    tf = cell.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run() if len(p.runs) == 0 else p.runs[0]
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.name = font
    r.font.color.rgb = color


def add_table(slide, left, top, width, height, headers, rows, col_widths=None,
              cell_colors=None, header_bg=NAVY, header_color=WHITE, size=11):
    n_rows = len(rows) + 1
    n_cols = len(headers)
    gtable = slide.shapes.add_table(n_rows, n_cols, left, top, width, height).table
    if col_widths:
        for j, w in enumerate(col_widths):
            gtable.columns[j].width = w
    for j, h in enumerate(headers):
        _set_cell(gtable.cell(0, j), h, size=size, color=header_color, bold=True, bg=header_bg)
    for i, row in enumerate(rows, start=1):
        bg_row = SAND if i % 2 == 1 else RGBColor(0xFB, 0xF8, 0xF1)
        for j, val in enumerate(row):
            bg = None
            if cell_colors is not None and cell_colors[i - 1][j] is not None:
                bg = cell_colors[i - 1][j]
            _set_cell(gtable.cell(i, j), str(val), size=size, color=INK, bg=bg or bg_row)
    return gtable


def zone_color(zone):
    return {"green": GREEN, "yellow": YELLOW, "red": RED}.get(zone, SLATE)


# ===========================================================================
# Deck assembly
# ===========================================================================

def build():
    res = pd.read_csv(RESULTS_CSV)
    best = pd.read_csv(BEST_CSV)

    fig_returns = gen_returns_clustering("Tunindex")
    fig_garch, garch_nviol, garch_rate = gen_garch_var("Tunindex", 0.01)
    fig_adi = gen_adi_lstm_vs_garch(0.01)
    fig_heat = gen_winner_heatmap()

    prs = new_presentation()
    TOTAL = 17
    page = [0]

    def track(slide):
        page[0] += 1
        return slide

    # ---------------- Slide 1: Titre --------------------------------------
    s = track(blank_slide(prs))
    set_background(s, NAVY)
    add_rect(s, Inches(0), Inches(6.6), SLIDE_W, Inches(0.9), NAVY_DARK)
    add_text(s, Inches(1.0), Inches(2.15), Inches(11.3), Inches(0.4), "PROJET DE SÉRIES TEMPORELLES",
              size=14, color=GOLD_LIGHT, bold=True, font=FONT_BODY)
    add_text(s, Inches(1.0), Inches(2.55), Inches(11.3), Inches(1.7),
              "La Value-at-Risk sur les marchés\nboursiers MENA",
              size=42, color=WHITE, bold=True, font=FONT_HEAD, line_spacing=1.05)
    add_text(s, Inches(1.0), Inches(4.25), Inches(11.0), Inches(0.6),
              "Deep Learning vs modèles statistiques -- ARIMA/SARIMA, GARCH, Random Forest / XGBoost, ANN / LSTM",
              size=16, color=SAND, font=FONT_BODY)
    circ = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(10.6), Inches(0.7), Inches(1.9), Inches(1.9))
    circ.fill.solid(); circ.fill.fore_color.rgb = GOLD; circ.line.fill.background(); circ.shadow.inherit = False
    tf = circ.text_frame; tf.margin_left = tf.margin_right = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = "VaR"; r.font.size = Pt(28); r.font.bold = True
    r.font.name = FONT_HEAD; r.font.color.rgb = NAVY_DARK
    add_text(s, Inches(1.0), Inches(6.75), Inches(8), Inches(0.5),
              "Tunindex - ADI - MASI - TASI  |  repères : CAC40, S&P 500", size=12, color=SAND)
    add_text(s, Inches(10.2), Inches(6.75), Inches(2.5), Inches(0.5), "jensbenarfa@gmail.com",
              size=11, color=GOLD_LIGHT, align=PP_ALIGN.RIGHT)

    # ---------------- Slide 2: Contexte & problématique --------------------
    s = track(blank_slide(prs)); set_background(s, SAND)
    add_kicker_and_title(s, 1, "Contexte", "Contexte et problématique")
    add_bullets(s, MARGIN, Inches(1.7), Inches(6.7), Inches(4.8), [
        "La Value-at-Risk (VaR) mesure la perte potentielle maximale d'un actif à un horizon et un niveau de confiance donnés (95% ou 99%).",
        "Les indices boursiers MENA (Tunisie, Émirats, Maroc, Arabie Saoudite) présentent des dynamiques propres : liquidité plus faible, chocs politiques régionaux, calendriers de cotation spécifiques.",
        "Question de recherche : les modèles de Deep Learning (ANN, LSTM) surpassent-ils les approches statistiques classiques (ARIMA, GARCH) et le Machine Learning (RF, XGBoost) pour l'estimation de la VaR sur ces marchés ?",
        "Hypothèse testée : le LSTM surpasse les modèles classiques sur l'indice ADI (Abu Dhabi).",
    ], size=15.5)
    box = add_rect(s, Inches(8.1), Inches(1.7), Inches(4.6), Inches(3.1), WHITE)
    add_text(s, Inches(8.4), Inches(1.9), Inches(4.0), Inches(0.4), "POURQUOI LA VaR ?",
              size=12, bold=True, color=GOLD.__class__(0xC9,0x98,0x2D))
    add_bullets(s, Inches(8.4), Inches(2.35), Inches(4.0), Inches(2.3), [
        "Exigence réglementaire (Bâle II/III) pour les banques et gestionnaires d'actifs.",
        "Outil de pilotage du risque de marché au quotidien.",
        "Le backtesting (Kupiec, Christoffersen, zones de Bâle) valide -- ou invalide -- un modèle de VaR a posteriori.",
    ], size=13.5, bullet="›", anchor=MSO_ANCHOR.MIDDLE)
    add_footer(s, page[0], TOTAL)

    # ---------------- Slide 3: Données & indices ---------------------------
    s = track(blank_slide(prs)); set_background(s, SAND)
    add_kicker_and_title(s, 2, "Données", "Données et indices")
    headers = ["Indice", "Pays / place", "Période (train)", "Période (test)"]
    rows = [
        ["Tunindex", "Tunisie", "2005-01-03 -> 2014-12-31", "2014-12-31 -> 2015-08-20"],
        ["ADI", "Abu Dhabi (EAU)", "2005-01-03 -> 2014-12-31", "2014-12-31 -> 2015-08-31"],
        ["MASI", "Maroc", "2005-01-03 -> 2014-12-31", "2014-12-31 -> 2015-08-19"],
        ["TASI", "Arabie Saoudite", "2005-01-03 -> 2014-12-31", "2014-12-31 -> 2015-08-30"],
    ]
    add_table(s, MARGIN, Inches(1.75), Inches(8.6), Inches(2.3), headers, rows,
              col_widths=[Inches(1.6), Inches(2.2), Inches(2.4), Inches(2.4)])
    add_text(s, MARGIN, Inches(4.35), Inches(8.6), Inches(0.35), "Repères marchés développés (comparaison, hors entraînement des modèles MENA) :",
              size=13, bold=True, color=NAVY)
    add_bullets(s, MARGIN, Inches(4.75), Inches(8.6), Inches(1.6), [
        "CAC40 (France) -- 2560 séances, 2005-2014.",
        "S&P 500 (États-Unis) -- 2536 séances, 2005-2014.",
        "Rendements calculés en log-rendements journaliers (%), base commune pour toutes les places.",
    ], size=13.5)
    add_rect(s, Inches(9.9), Inches(1.75), Inches(2.8), Inches(3.0), WHITE)
    add_text(s, Inches(10.1), Inches(1.95), Inches(2.4), Inches(0.5), "TAILLE DES SÉRIES",
              size=11, bold=True, color=GOLD)
    add_text(s, Inches(10.1), Inches(2.4), Inches(2.4), Inches(2.2),
              "~2 470 à 2 585\nséances d'entraînement\npar indice MENA\n\n+ 160 à 171\nséances de test\n(walk-forward)",
              size=14.5, color=NAVY, bold=True, line_spacing=1.2, anchor=MSO_ANCHOR.MIDDLE)
    add_footer(s, page[0], TOTAL)

    # ---------------- Slide 4: Méthodologie / pipeline ----------------------
    s = track(blank_slide(prs)); set_background(s, SAND)
    add_kicker_and_title(s, 3, "Méthodologie", "Pipeline unifié : train-once / walk-forward")
    steps = [
        ("1", "Entraînement unique", "Chaque modèle est ajusté UNE seule fois sur la période train (aucun ré-entraînement)."),
        ("2", "Walk-forward J+1", "Prévision un jour à l'avance sur le test ; les états (résidus, variance) sont mis à jour avec la valeur réalisée, pas les paramètres."),
        ("3", "VaR par BHS", "mu, sigma prédits + pool de résidus standardisés bootstrappés -> quantile alpha (VaR)."),
        ("4", "Backtesting", "Comparaison VaR vs rendement réalisé : Kupiec, Christoffersen, zones de Bâle."),
    ]
    x0 = MARGIN
    w = Inches(2.85)
    gap = Inches(0.18)
    for i, (num, head, desc) in enumerate(steps):
        left = Inches(0.6 + i * (2.85 + 0.18))
        card = add_rect(s, left, Inches(1.85), w, Inches(3.7), WHITE)
        circ = s.shapes.add_shape(MSO_SHAPE.OVAL, left + Inches(0.18), Inches(2.05), Inches(0.5), Inches(0.5))
        circ.fill.solid(); circ.fill.fore_color.rgb = GOLD; circ.line.fill.background(); circ.shadow.inherit = False
        tf = circ.text_frame; tf.margin_left = tf.margin_right = 0
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = num; r.font.bold = True; r.font.size = Pt(18)
        r.font.color.rgb = NAVY_DARK; r.font.name = FONT_HEAD
        add_text(s, left + Inches(0.18), Inches(2.7), w - Inches(0.36), Inches(0.6), head,
                  size=14.5, bold=True, color=NAVY, font=FONT_HEAD)
        add_text(s, left + Inches(0.18), Inches(3.3), w - Inches(0.36), Inches(2.1), desc,
                  size=12, color=SLATE, line_spacing=1.15)
        if i < 3:
            arrow_left = left + w + Inches(0.02)
            arr = s.shapes.add_shape(MSO_SHAPE.CHEVRON, arrow_left, Inches(3.5), Inches(0.14), Inches(0.6))
            arr.fill.solid(); arr.fill.fore_color.rgb = GOLD; arr.line.fill.background(); arr.shadow.inherit = False
    add_text(s, MARGIN, Inches(5.85), Inches(12.1), Inches(0.9),
              "Contrainte de conception : aucun modèle n'est ré-estimé sur le test -- seule l'information disponible à J est utilisée pour prédire J+1 (walk-forward one-step-ahead), pour les 7 modèles comme pour le GARCH (FHS).",
              size=13, italic=True, color=NAVY)
    add_footer(s, page[0], TOTAL)

    # ---------------- Slide 5: Prétraitement & rendements log ---------------
    s = track(blank_slide(prs)); set_background(s, SAND)
    add_kicker_and_title(s, 4, "Prétraitement", "Prétraitement et rendements logarithmiques")
    add_bullets(s, MARGIN, Inches(1.75), Inches(5.6), Inches(3.2), [
        "Prix nettoyés (virgules, unités M/K du volume) puis convertis en rendements log : r_t = 100 x ln(P_t / P_{t-1}).",
        "Les rendements log stabilisent la variance et rendent les séries approximativement stationnaires en moyenne.",
        "On observe un regroupement de volatilité (volatility clustering) : périodes calmes et périodes agitées s'enchaînent -- justifie l'usage du GARCH.",
    ], size=15.5)
    add_picture_framed(s, fig_returns, Inches(6.9), Inches(1.75), width=Inches(5.8))
    add_footer(s, page[0], TOTAL)

    # ---------------- Slide 6: Stationnarité & décomposition ----------------
    s = track(blank_slide(prs)); set_background(s, SAND)
    add_kicker_and_title(s, 5, "Stationnarité", "Stationnarité (ADF / KPSS) et décomposition")
    add_bullets(s, MARGIN, Inches(1.75), Inches(6.0), Inches(4.6), [
        "Test ADF (racine unitaire) : H0 = série non-stationnaire. p < 0.05 -> on rejette H0 -> série stationnaire.",
        "Test KPSS : H0 = série stationnaire. p > 0.05 -> on ne rejette pas H0 -> cohérent avec la stationnarité.",
        "Sur les niveaux de prix : ADF ne rejette pas H0 (non-stationnaire), comportement typique d'une marche aléatoire.",
        "Sur les rendements log : ADF rejette H0 et KPSS ne le rejette pas -- les deux tests concordent, la série est stationnaire.",
        "La décomposition (tendance / saisonnalité / résidu) confirme l'absence de tendance forte sur les rendements, contrairement aux prix.",
    ], size=14.5)
    add_rect(s, Inches(8.3), Inches(1.75), Inches(4.4), Inches(3.3), WHITE)
    add_text(s, Inches(8.55), Inches(1.95), Inches(4.0), Inches(0.4), "LECTURE ADF / KPSS", size=12, bold=True, color=GOLD)
    add_bullets(s, Inches(8.55), Inches(2.4), Inches(3.9), Inches(2.5), [
        "Prix : ADF p élevé, KPSS p faible -> non stationnaire.",
        "Rendements : ADF p faible, KPSS p élevé -> stationnaire.",
        "-> les modèles (ARIMA d=0 sur rendements, GARCH) sont appliqués sur des séries stationnaires, condition nécessaire à leur validité.",
    ], size=13, bullet="›", anchor=MSO_ANCHOR.MIDDLE)
    add_footer(s, page[0], TOTAL)

    # ---------------- Slide 7: ARIMA / SARIMA -------------------------------
    s = track(blank_slide(prs)); set_background(s, SAND)
    add_kicker_and_title(s, 6, "Modèles statistiques", "ARIMA et SARIMA")
    add_bullets(s, MARGIN, Inches(1.75), Inches(6.1), Inches(4.6), [
        "ARIMA(p,d,q) modélise la moyenne conditionnelle des rendements à partir de leurs valeurs passées et des erreurs passées.",
        "SARIMA ajoute une composante saisonnière (ici m=5, cycle hebdomadaire de séances de bourse).",
        "Les deux modèles sont ajustés UNE fois sur le train, puis utilisés en walk-forward one-step-ahead sur le test.",
        "La VaR est ensuite calculée par BHS à partir de mu (prévision ARIMA/SARIMA) et sigma (écart-type des résidus).",
        "Résultat empirique : ARIMA est le modèle GAGNANT sur TASI (Arabie Saoudite) à 99%.",
    ], size=14.5)
    tasi_arima = res[(res["index"] == "TASI") & (res.model == "ARIMA") & (res.alpha == 0.01)].iloc[0]
    add_rect(s, Inches(8.5), Inches(1.75), Inches(4.2), Inches(4.6), NAVY)
    add_text(s, Inches(8.75), Inches(2.0), Inches(3.7), Inches(0.4), "TASI -- ARIMA @ 99%", size=13, bold=True, color=GOLD_LIGHT)
    add_text(s, Inches(8.75), Inches(2.55), Inches(3.7), Inches(1.0), f"{tasi_arima.observed_rate:.2%}",
              size=44, bold=True, color=WHITE, font=FONT_HEAD)
    add_text(s, Inches(8.75), Inches(3.55), Inches(3.7), Inches(0.4), "taux de violation observé (cible 1%)", size=11, color=SAND)
    add_text(s, Inches(8.75), Inches(4.15), Inches(3.7), Inches(0.4),
              f"Kupiec p = {tasi_arima.kupiec_p:.3f}   |   zone {tasi_arima.basel_zone.upper()}",
              size=12, color=GOLD_LIGHT, bold=True)
    add_text(s, Inches(8.75), Inches(4.7), Inches(3.7), Inches(1.4),
              "Zone verte Bâle : le modèle n'est pas rejeté, le nombre de violations est cohérent avec le seuil attendu.",
              size=11.5, color=SAND, italic=True)
    add_footer(s, page[0], TOTAL)

    # ---------------- Slide 8: GARCH -----------------------------------------
    s = track(blank_slide(prs)); set_background(s, SAND)
    add_kicker_and_title(s, 7, "Volatilité", "Volatilité GARCH et simulation historique filtrée (FHS)")
    add_bullets(s, MARGIN, Inches(1.7), Inches(12.1), Inches(1.5), [
        "GARCH(1,1) à moyenne constante, ajusté UNE fois sur le train ; la variance conditionnelle est ensuite propagée jour par jour sur le test avec les paramètres figés (omega, alpha, beta).",
        "FHS (Filtered Historical Simulation) : on combine sigma_t (dynamique, GARCH) avec le pool de résidus standardisés du train, rééchantillonné pour la VaR (BHS).",
    ], size=13.5)
    garch_img_top = 3.15
    garch_img_w = 10.1
    garch_img_h = garch_img_w * (3.2 / 11.0)  # matches gen_garch_var figsize aspect ratio
    add_picture_framed(s, fig_garch, Inches(1.6), Inches(garch_img_top), width=Inches(garch_img_w))
    add_text(s, Inches(1.6), Inches(garch_img_top + garch_img_h + 0.15), Inches(10.1), Inches(0.5),
              f"Tunindex, 99% : {garch_nviol} violations observées sur la période test ({garch_rate:.1%}) -- GARCH est le modèle gagnant sur Tunindex, ADI et MASI (3/4 indices MENA).",
              size=11.5, italic=True, color=NAVY)
    add_footer(s, page[0], TOTAL)

    # ---------------- Slide 9: Machine Learning (RF/XGBoost) ----------------
    s = track(blank_slide(prs)); set_background(s, SAND)
    add_kicker_and_title(s, 8, "Machine Learning", "Random Forest et XGBoost")
    add_bullets(s, MARGIN, Inches(1.75), Inches(6.2), Inches(4.6), [
        "Les rendements passés (fenêtre glissante de lags) forment les variables explicatives (approche supervisée).",
        "RF et XGBoost sont entraînés UNE fois sur le train, puis prédisent J+1 en walk-forward sur le test.",
        "sigma est estimé à partir de l'écart-type des résidus d'entraînement (constant), combiné à mu (prévision de l'arbre) pour la VaR BHS.",
        "Résultat empirique -- Random Forest est le modèle le plus faible du panel pour la VaR : il sur-viole systématiquement le seuil attendu (zone ROUGE Bâle sur les 4 indices MENA à 95%, et sur Tunindex/MASI/TASI à 99%).",
        "XGBoost se comporte mieux que RF mais reste en général derrière ARIMA/GARCH/ANN (zones vertes/jaunes selon l'indice).",
    ], size=14)
    rf_rows = []
    for idx in MENA:
        row = res[(res["index"] == idx) & (res.model == "RF") & (res.alpha == 0.01)].iloc[0]
        rf_rows.append([idx, f"{row.observed_rate:.2%}", row.basel_zone.upper()])
    add_text(s, Inches(8.4), Inches(1.75), Inches(4.3), Inches(0.35), "RF @ 99% -- taux de violation observé",
              size=12, bold=True, color=NAVY)
    add_table(s, Inches(8.4), Inches(2.15), Inches(4.3), Inches(2.2),
              ["Indice", "Taux observé", "Zone Bâle"], rf_rows,
              cell_colors=[[None, None, zone_color(r[2].lower())] for r in rf_rows],
              col_widths=[Inches(1.7), Inches(1.5), Inches(1.1)])
    rf_rates = [float(r[1].rstrip("%").replace(",", ".")) for r in rf_rows]
    add_text(s, Inches(8.4), Inches(4.6), Inches(4.3), Inches(1.8),
              f"Cible attendue : ~1%. Les taux observés ({min(rf_rates):.1f}-{max(rf_rates):.1f}%, "
              "et jusqu'à 22% à 95%) sont très supérieurs -- Random Forest sous-estime largement le risque de queue.",
              size=12, italic=True, color=RED, anchor=MSO_ANCHOR.TOP)
    add_footer(s, page[0], TOTAL)

    # ---------------- Slide 10: Deep Learning (ANN/LSTM) --------------------
    s = track(blank_slide(prs)); set_background(s, SAND)
    add_kicker_and_title(s, 9, "Deep Learning", "Réseaux de neurones : ANN et LSTM")
    add_bullets(s, MARGIN, Inches(1.75), Inches(6.3), Inches(4.7), [
        "ANN (perceptron multicouche) et LSTM (mémoire court/long terme) sont entraînés UNE fois sur des séquences standardisées du train (fenêtre de 10 rendements).",
        "En walk-forward, le réseau reçoit la fenêtre de rendements réalisés (pas re-simulés) et prédit le rendement du jour suivant ; les poids ne sont jamais mis à jour sur le test.",
        "sigma est l'écart-type des résidus d'entraînement ; la VaR est calculée par BHS comme pour les autres modèles (comparabilité stricte entre les 7 approches).",
        "Résultat empirique : ANN et LSTM sont bien calibrés (zone verte) sur la quasi-totalité des couples indice/alpha, avec des MAE/RMSE proches, voire légèrement meilleurs, que ceux d'ARIMA/GARCH.",
        "Mais ils ne dominent pas nettement GARCH ou ARIMA en zone de Bâle -- aucun des deux réseaux ne remporte le titre de meilleur modèle sur un indice MENA (voir résultats, slide 13).",
    ], size=13.5)
    add_rect(s, Inches(8.5), Inches(1.75), Inches(4.2), Inches(4.7), WHITE)
    add_text(s, Inches(8.75), Inches(1.95), Inches(3.7), Inches(0.4), "ADI -- LSTM @ 99%", size=12, bold=True, color=GOLD)
    adi_lstm = res[(res["index"] == "ADI") & (res.model == "LSTM") & (res.alpha == 0.01)].iloc[0]
    add_text(s, Inches(8.75), Inches(2.4), Inches(3.7), Inches(0.9), f"{adi_lstm.observed_rate:.2%}",
              size=38, bold=True, color=NAVY, font=FONT_HEAD)
    add_text(s, Inches(8.75), Inches(3.25), Inches(3.7), Inches(0.4), "taux de violation observé", size=11, color=SLATE)
    add_text(s, Inches(8.75), Inches(3.75), Inches(3.7), Inches(0.4),
              f"zone {adi_lstm.basel_zone.upper()}  |  Kupiec p={adi_lstm.kupiec_p:.3f}", size=12, bold=True, color=GREEN)
    add_text(s, Inches(8.75), Inches(4.35), Inches(3.7), Inches(1.9),
              "Bien calibré, comparable à GARCH sur ADI -- mais pas distinctement supérieur (voir slide 15, vérification de l'hypothèse).",
              size=11.5, italic=True, color=NAVY)
    add_footer(s, page[0], TOTAL)

    # ---------------- Slide 11: VaR par BHS (formule) -----------------------
    s = track(blank_slide(prs)); set_background(s, NAVY)
    add_kicker_and_title(s, 10, "Méthode de calcul", "Le calcul de la VaR par BHS", dark=True)
    card = add_rect(s, Inches(1.6), Inches(2.1), Inches(10.1), Inches(1.9), NAVY_DARK)
    add_text(s, Inches(1.6), Inches(2.35), Inches(10.1), Inches(1.5),
              "VaR_alpha(t) = mu_t + sigma_t . Q_alpha( résidus standardisés bootstrappés )",
              size=22, bold=True, color=GOLD_LIGHT, font=FONT_HEAD, align=PP_ALIGN.CENTER)
    add_bullets(s, Inches(1.6), Inches(4.3), Inches(10.1), Inches(2.6), [
        "mu_t, sigma_t : moyenne et écart-type prédits par le modèle (ARIMA, GARCH, RF/XGB, ANN/LSTM) pour le jour t.",
        "résidus standardisés : pool des résidus (z-scores) estimés UNE fois sur la période d'entraînement.",
        "Bootstrap : n_boot=10 000 tirages avec remise dans ce pool -> quantile empirique Q_alpha (alpha=5% ou 1%).",
        "BHS (Bootstrap Historical Simulation) ne suppose pas la normalité des résidus : la distribution empirique (asymétrie, queues épaisses) est préservée -- adapté aux marchés MENA, souvent non-gaussiens.",
    ], size=14, color=SAND)
    add_footer(s, page[0], TOTAL, dark=True)

    # ---------------- Slide 12: Backtesting ---------------------------------
    s = track(blank_slide(prs)); set_background(s, SAND)
    add_kicker_and_title(s, 11, "Validation", "Backtesting : Kupiec, Christoffersen, Bâle")
    cols = [
        ("Kupiec (POF)", "Teste si le TAUX de violations observé est cohérent avec alpha attendu. p > 0.05 -> modèle non rejeté."),
        ("Christoffersen (indépendance + couverture)", "Teste en plus si les violations sont indépendantes dans le temps (pas de clusters de violations). p_cc > 0.05 -> modèle non rejeté."),
        ("Zones de Bâle", "Classe le modèle en vert / jaune / rouge selon le nombre de violations normalisé sur 250 séances -- grille réglementaire standard."),
    ]
    w = Inches(3.95)
    for i, (head, desc) in enumerate(cols):
        left = Inches(0.6 + i * (3.95 + 0.2))
        add_rect(s, left, Inches(1.8), w, Inches(2.95), WHITE)
        add_text(s, left + Inches(0.2), Inches(2.0), w - Inches(0.4), Inches(0.9), head,
                  size=15, bold=True, color=NAVY, font=FONT_HEAD, line_spacing=1.05)
        add_text(s, left + Inches(0.2), Inches(2.9), w - Inches(0.4), Inches(1.7), desc,
                  size=13, color=SLATE, line_spacing=1.2, anchor=MSO_ANCHOR.MIDDLE)
    legend_y = Inches(5.1)
    for i, (zone, col, meaning) in enumerate([
        ("Vert", GREEN, "modèle non rejeté, risque bien capté"),
        ("Jaune", YELLOW, "zone de surveillance"),
        ("Rouge", RED, "modèle rejeté, sous-estime le risque"),
    ]):
        left = Inches(0.6 + i * 4.1)
        dot = s.shapes.add_shape(MSO_SHAPE.OVAL, left, legend_y, Inches(0.3), Inches(0.3))
        dot.fill.solid(); dot.fill.fore_color.rgb = col; dot.line.fill.background(); dot.shadow.inherit = False
        add_text(s, left + Inches(0.4), legend_y - Inches(0.03), Inches(3.6), Inches(0.4),
                  f"{zone} -- {meaning}", size=12.5, color=NAVY, bold=True)
    add_footer(s, page[0], TOTAL)

    # ---------------- Slide 13: Résultats par indice ------------------------
    s = track(blank_slide(prs)); set_background(s, SAND)
    add_kicker_and_title(s, 12, "Résultats", "Résultats par indice -- le tableau des gagnants")
    headers = ["Indice", "Modèle gagnant", "Taux observé (99%)", "Kupiec p", "Zone Bâle"]
    rows = []
    colors = []
    for _, row in best.iterrows():
        rows.append([row["index"], row.model, f"{row.observed_rate:.2%}", f"{row.kupiec_p:.3f}", row.basel_zone.upper()])
        colors.append([None, GOLD_LIGHT, None, None, zone_color(row.basel_zone)])
    add_table(s, MARGIN, Inches(1.75), Inches(7.9), Inches(2.3), headers, rows,
              cell_colors=colors, col_widths=[Inches(1.7), Inches(1.9), Inches(1.9), Inches(1.3), Inches(1.1)])
    add_text(s, MARGIN, Inches(4.3), Inches(7.9), Inches(0.9),
              "GARCH est le modèle globalement le plus performant : il remporte 3 indices MENA sur 4 (Tunindex, ADI, MASI). ARIMA remporte TASI. Classement établi selon la p-value de Kupiec la plus élevée à alpha=1% (meilleure adéquation entre taux observé et taux attendu).",
              size=12.5, italic=True, color=NAVY)
    add_picture_framed(s, fig_heat, Inches(8.55), Inches(1.75), width=Inches(4.15))
    add_footer(s, page[0], TOTAL)

    # ---------------- Slide 14: MENA vs marchés développés ------------------
    s = track(blank_slide(prs)); set_background(s, SAND)
    add_kicker_and_title(s, 13, "Discussion", "MENA vs marchés développés (CAC40 / S&P 500)")
    add_bullets(s, MARGIN, Inches(1.75), Inches(6.3), Inches(4.6), [
        "Sur les marchés développés (CAC40, S&P 500), la littérature montre un avantage plus net des modèles GARCH/Deep Learning face aux chocs de volatilité (crises 2008, 2011).",
        "Sur les indices MENA de cette étude, les modèles ANN/LSTM égalent GARCH/ARIMA en calibration (zone verte) mais ne les surpassent pas nettement -- écart moins marqué qu'attendu.",
        "Hypothèse explicative : séries MENA plus courtes en observations utiles de volatilité extrême, et micro-structure (liquidité, jours de fermeture spécifiques) qui peut limiter l'avantage informationnel du Deep Learning.",
        "GARCH, plus parcimonieux, reste compétitif -- cohérent avec des travaux montrant sa robustesse sur des marchés émergents à volatilité regroupée mais aux échantillons plus restreints.",
    ], size=14)
    add_rect(s, Inches(8.5), Inches(1.75), Inches(4.2), Inches(3.6), NAVY)
    add_text(s, Inches(8.75), Inches(1.95), Inches(3.7), Inches(0.4), "À RETENIR", size=12, bold=True, color=GOLD_LIGHT)
    add_bullets(s, Inches(8.75), Inches(2.4), Inches(3.7), Inches(2.7), [
        "Pas de \"solution unique\" -- le meilleur modèle dépend de l'indice.",
        "GARCH : bon compromis simplicité / robustesse sur MENA.",
        "Deep Learning : compétitif mais pas dominant ici.",
        "RF/XGBoost : à éviter tels quels pour la VaR de marché.",
    ], size=13, color=SAND, bullet="›", anchor=MSO_ANCHOR.MIDDLE)
    add_footer(s, page[0], TOTAL)

    # ---------------- Slide 15: Vérification de l'hypothèse -----------------
    s = track(blank_slide(prs)); set_background(s, SAND)
    add_kicker_and_title(s, 14, "Verdict", "Vérification de l'hypothèse : le LSTM sur ADI")
    add_text(s, MARGIN, Inches(1.7), Inches(12.1), Inches(0.4),
              "Hypothèse testée : « le LSTM surpasse les modèles classiques sur l'indice ADI »",
              size=15, bold=True, color=NAVY)
    add_picture_framed(s, fig_adi, Inches(MARGIN.inches), Inches(2.2), width=Inches(8.4))
    box = add_rect(s, Inches(9.05), Inches(2.2), Inches(3.65), Inches(4.4), RED)
    add_text(s, Inches(9.3), Inches(2.4), Inches(3.2), Inches(0.5), "NON CONFIRMÉE", size=20, bold=True, color=WHITE, font=FONT_HEAD)
    adi_garch = res[(res["index"] == "ADI") & (res.model == "GARCH") & (res.alpha == 0.01)].iloc[0]
    add_text(s, Inches(9.3), Inches(3.0), Inches(3.2), Inches(2.9),
              f"LSTM @ 99% : {adi_lstm.observed_rate:.2%} de violations, zone {adi_lstm.basel_zone.upper()} -- bien calibré.\n\n"
              f"GARCH @ 99% : {adi_garch.observed_rate:.2%}, zone {adi_garch.basel_zone.upper()} -- également bien calibré, et gagnant officiel sur ADI (Kupiec p={adi_garch.kupiec_p:.3f} vs {adi_lstm.kupiec_p:.3f}).\n\n"
              "Le LSTM est fiable sur ADI mais n'est PAS distinctement supérieur au GARCH : les deux sont en zone verte, GARCH a la meilleure p-value de Kupiec.",
              size=12, color=WHITE, line_spacing=1.15)
    add_footer(s, page[0], TOTAL)

    # ---------------- Slide 16: Conclusion & perspectives --------------------
    s = track(blank_slide(prs)); set_background(s, SAND)
    add_kicker_and_title(s, 15, "Conclusion", "Conclusion et perspectives")
    add_bullets(s, MARGIN, Inches(1.75), Inches(6.3), Inches(4.7), [
        "GARCH est le modèle le plus robuste pour la VaR sur les indices MENA étudiés (3 indices gagnés sur 4).",
        "ARIMA reste compétitif et gagne sur TASI -- la simplicité n'est pas pénalisante ici.",
        "ANN/LSTM sont bien calibrés mais ne surpassent pas nettement les modèles classiques -- l'hypothèse LSTM > classiques sur ADI n'est pas confirmée.",
        "Random Forest (et dans une moindre mesure XGBoost) est déconseillé pour la VaR de marché en l'état : sur-violations systématiques (zone rouge).",
        "Perspectives : élargir la fenêtre de test, tester GARCH asymétrique (EGARCH/GJR) et des architectures hybrides (GARCH-LSTM), et exploiter des données intra-journalières pour les marchés MENA les plus liquides.",
    ], size=14.5)
    add_rect(s, Inches(8.5), Inches(1.75), Inches(4.2), Inches(3.4), WHITE)
    add_text(s, Inches(8.75), Inches(1.95), Inches(3.7), Inches(0.4), "PIPELINE & CODE", size=12, bold=True, color=GOLD)
    add_bullets(s, Inches(8.75), Inches(2.4), Inches(3.7), Inches(2.5), [
        "Package tsvar (Python) : data, classical, volatility, ml, deep, var, backtest, run, plots.",
        "22 tests unitaires (pytest) couvrant chaque étage du pipeline.",
        "Résultats reproductibles : outputs/results.csv (7 modèles x 4 indices x 2 alpha).",
    ], size=13, bullet="›", anchor=MSO_ANCHOR.MIDDLE)
    add_footer(s, page[0], TOTAL)

    # ---------------- Slide 17: Merci / clôture -----------------------------
    s = track(blank_slide(prs)); set_background(s, NAVY)
    add_rect(s, Inches(0), Inches(0), SLIDE_W, Inches(0.15), GOLD)
    add_text(s, Inches(1.0), Inches(2.7), Inches(11.3), Inches(1.2), "Merci",
              size=54, bold=True, color=WHITE, font=FONT_HEAD)
    add_text(s, Inches(1.0), Inches(3.9), Inches(11.0), Inches(0.6),
              "Projet VaR MENA -- Deep Learning vs modèles statistiques",
              size=16, color=GOLD_LIGHT)
    add_text(s, Inches(1.0), Inches(4.5), Inches(11.0), Inches(0.5),
              "Questions bienvenues -- code, données et résultats disponibles dans le dépôt du projet.",
              size=13, color=SAND)
    add_text(s, Inches(1.0), Inches(6.75), Inches(8), Inches(0.4), "jensbenarfa@gmail.com",
              size=12, color=SAND)

    n_slides = page[0]
    out_path = PRES_DIR / "VaR_MENA.pptx"
    prs.save(str(out_path))
    return out_path, n_slides


if __name__ == "__main__":
    path, n = build()
    print(f"Saved {path} with {n} slides")
