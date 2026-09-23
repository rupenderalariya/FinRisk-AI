"""
Visual theme: palette, CSS and the shared Plotly template.

All styling lives here. No other module defines a colour or writes CSS, so the
product stays visually consistent and a change to the palette propagates
everywhere at once.

Design direction
----------------
A dark financial-intelligence aesthetic: deep background, slightly raised
surface panels, hairline borders, strong numeric typography, and restrained
accent use. Colour carries *meaning* rather than decoration:

===========  ==========================================
neutral      normal, no judgement implied
blue         information and the primary series
green        positive / opportunity
amber        attention / watch
red          risk
===========  ==========================================

Charts never pick arbitrary colours; they draw from this palette so a reader can
learn the colour language once and rely on it.

CSS caution
-----------
Selectors target stable Streamlit test IDs (``data-testid``) and our own ``fr-``
prefixed classes. Nothing here overrides Streamlit's interaction behaviour, so
widgets keep working normally.
"""

from __future__ import annotations

from typing import Final, Mapping

import plotly.graph_objects as go
import plotly.io as pio

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #

PALETTE: Final[Mapping[str, str]] = {
    # --- structural ---
    "bg": "#0a0e14",
    "bg_alt": "#0d1218",
    "surface": "#121922",
    "surface_raised": "#18212c",
    "surface_hover": "#1d2735",
    "border": "#222d3b",
    "border_strong": "#2e3c4d",
    # --- text ---
    "text": "#e8eef6",
    "text_secondary": "#a8b6c7",
    "text_muted": "#6f7f92",
    "text_faint": "#4a5768",
    # --- semantic ---
    "info": "#4d94ff",
    "info_dim": "#2d5f9e",
    "positive": "#16b47f",
    "positive_dim": "#0f6b4d",
    "attention": "#e8a33d",
    "attention_dim": "#8f6320",
    "risk": "#ef5f5f",
    "risk_dim": "#8f3434",
    "neutral": "#64748b",
    "accent": "#7c6cf0",
}

#: Severity name -> colour. Used by risk signals and quality badges.
SEVERITY_COLOURS: Final[Mapping[str, str]] = {
    "critical": PALETTE["risk"],
    "high": PALETTE["risk"],
    "medium": PALETTE["attention"],
    "low": PALETTE["info"],
    "info": PALETTE["neutral"],
}

#: Finding type -> colour.
FINDING_COLOURS: Final[Mapping[str, str]] = {
    "risk": PALETTE["risk"],
    "opportunity": PALETTE["positive"],
    "observation": PALETTE["info"],
    "data_quality": PALETTE["attention"],
}

#: Qualitative sequence for multi-series charts. Ordered so the first three stay
#: distinguishable under the common forms of colour-vision deficiency.
CATEGORICAL_SEQUENCE: Final[tuple[str, ...]] = (
    "#4d94ff",
    "#e8a33d",
    "#16b47f",
    "#7c6cf0",
    "#ef5f5f",
    "#3fbfc4",
    "#b47fd6",
    "#8b98a9",
)

#: Single-hue sequential scale for ordered magnitude on a dark surface.
SEQUENTIAL_SCALE: Final[tuple[tuple[float, str], ...]] = (
    (0.00, "#111a24"),
    (0.25, "#1b3a52"),
    (0.50, "#255a7d"),
    (0.75, "#3480ae"),
    (1.00, "#4d94ff"),
)

#: Diverging scale for correlations: red-neutral-blue reads as
#: negative-none-positive and survives greyscale printing.
DIVERGING_SCALE: Final[tuple[tuple[float, str], ...]] = (
    (0.0, "#ef5f5f"),
    (0.5, "#1b2430"),
    (1.0, "#4d94ff"),
)

FONT_STACK: Final[str] = (
    'ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", '
    'Roboto, "Helvetica Neue", Arial, sans-serif'
)
MONO_STACK: Final[str] = (
    'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace'
)

PLOTLY_TEMPLATE_NAME: Final[str] = "finrisk_dark"


# --------------------------------------------------------------------------- #
# Plotly template
# --------------------------------------------------------------------------- #


def build_plotly_template() -> go.layout.Template:
    """Construct the shared dark Plotly template.

    Registering a template rather than styling each figure means every chart in
    the product inherits the same surface, grid, font and hover treatment.

    Returns:
        The template object.
    """
    return go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": FONT_STACK, "size": 12.5, "color": PALETTE["text_secondary"]},
            title={
                "font": {"family": FONT_STACK, "size": 15, "color": PALETTE["text"]},
                "x": 0,
                "xanchor": "left",
                "y": 0.97,
                "yanchor": "top",
            },
            colorway=list(CATEGORICAL_SEQUENCE),
            xaxis={
                "gridcolor": PALETTE["border"],
                "zerolinecolor": PALETTE["border_strong"],
                "linecolor": PALETTE["border"],
                "tickfont": {"size": 11.5, "color": PALETTE["text_muted"]},
                "title": {"font": {"size": 12, "color": PALETTE["text_secondary"]}},
                "showgrid": False,
                "ticks": "outside",
                "tickcolor": PALETTE["border"],
                "automargin": True,
            },
            yaxis={
                "gridcolor": PALETTE["border"],
                "zerolinecolor": PALETTE["border_strong"],
                "linecolor": PALETTE["border"],
                "tickfont": {"size": 11.5, "color": PALETTE["text_muted"]},
                "title": {"font": {"size": 12, "color": PALETTE["text_secondary"]}},
                "showgrid": True,
                "gridwidth": 1,
                "automargin": True,
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.0,
                "xanchor": "right",
                "x": 1.0,
                "bgcolor": "rgba(0,0,0,0)",
                "font": {"size": 11.5, "color": PALETTE["text_secondary"]},
            },
            hoverlabel={
                "bgcolor": PALETTE["surface_raised"],
                "bordercolor": PALETTE["border_strong"],
                "font": {"family": FONT_STACK, "size": 12, "color": PALETTE["text"]},
            },
            margin={"l": 58, "r": 22, "t": 52, "b": 46},
            separators=".,",
        )
    )


def register_plotly_template() -> None:
    """Register the template with Plotly and make it the default.

    Idempotent, which matters because Streamlit re-executes modules on every
    interaction.
    """
    pio.templates[PLOTLY_TEMPLATE_NAME] = build_plotly_template()
    pio.templates.default = PLOTLY_TEMPLATE_NAME


# --------------------------------------------------------------------------- #
# CSS
# --------------------------------------------------------------------------- #


def build_css() -> str:
    """Assemble the stylesheet.

    Returns:
        A ``<style>`` block ready to inject.
    """
    p = PALETTE
    return f"""
<style>
:root {{
  --fr-bg: {p['bg']};
  --fr-surface: {p['surface']};
  --fr-surface-raised: {p['surface_raised']};
  --fr-border: {p['border']};
  --fr-border-strong: {p['border_strong']};
  --fr-text: {p['text']};
  --fr-text-secondary: {p['text_secondary']};
  --fr-text-muted: {p['text_muted']};
  --fr-info: {p['info']};
  --fr-positive: {p['positive']};
  --fr-attention: {p['attention']};
  --fr-risk: {p['risk']};
  --fr-neutral: {p['neutral']};
  --fr-radius: 10px;
  --fr-gap: 14px;
}}

/* ---------- app shell ---------- */
.stApp, [data-testid="stAppViewContainer"] {{
  background: {p['bg']};
  color: var(--fr-text);
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stToolbar"] {{ right: 0.5rem; }}
[data-testid="stAppViewContainer"] > .main .block-container {{
  padding: 1.6rem 2.2rem 3.5rem 2.2rem;
  max-width: 1600px;
}}
#MainMenu, footer {{ visibility: hidden; }}

html, body, [class*="css"] {{
  font-family: {FONT_STACK};
  -webkit-font-smoothing: antialiased;
}}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{
  background: {p['bg_alt']};
  border-right: 1px solid var(--fr-border);
}}
[data-testid="stSidebar"] > div:first-child {{ padding-top: 1.1rem; }}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 0.28rem; }}

.fr-brand {{ padding: 0 0.35rem 0.9rem 0.35rem; }}
.fr-brand-row {{ display: flex; align-items: center; gap: 0.6rem; }}
.fr-brand-mark {{
  width: 32px; height: 32px; border-radius: 8px; flex: 0 0 32px;
  background: linear-gradient(135deg, {p['info']} 0%, {p['accent']} 100%);
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 13px; color: #fff; letter-spacing: -0.5px;
}}
.fr-brand-name {{
  font-size: 1.02rem; font-weight: 650; color: var(--fr-text);
  letter-spacing: -0.2px; line-height: 1.1;
}}
.fr-brand-sub {{
  font-size: 0.66rem; color: var(--fr-text-muted);
  text-transform: uppercase; letter-spacing: 0.7px; margin-top: 2px;
}}

.fr-nav-group {{
  font-size: 0.63rem; font-weight: 650; color: {p['text_faint']};
  text-transform: uppercase; letter-spacing: 1.1px;
  margin: 1.05rem 0 0.35rem 0.55rem;
}}

/* Sidebar nav buttons styled as list items rather than buttons. */
[data-testid="stSidebar"] .stButton > button {{
  width: 100%; text-align: left; justify-content: flex-start;
  background: transparent; color: var(--fr-text-secondary);
  border: 1px solid transparent; border-radius: 8px;
  padding: 0.44rem 0.65rem; font-size: 0.855rem; font-weight: 500;
  transition: background 120ms ease, color 120ms ease;
  box-shadow: none;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
  background: {p['surface']}; color: var(--fr-text); border-color: var(--fr-border);
}}
[data-testid="stSidebar"] .stButton > button:focus {{ box-shadow: none; outline: none; }}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
  background: {p['surface_raised']}; color: var(--fr-text);
  border-color: var(--fr-border-strong); font-weight: 600;
  box-shadow: inset 2px 0 0 0 {p['info']};
}}

/* ---------- page header ---------- */
.fr-page-head {{
  display: flex; align-items: flex-start; justify-content: space-between;
  gap: 2rem; padding-bottom: 0.9rem; margin-bottom: 1.3rem;
  border-bottom: 1px solid var(--fr-border);
}}
.fr-page-title {{
  font-size: 1.62rem; font-weight: 660; color: var(--fr-text);
  letter-spacing: -0.5px; line-height: 1.15; margin: 0;
}}
.fr-page-desc {{
  font-size: 0.875rem; color: var(--fr-text-muted);
  margin-top: 0.36rem; max-width: 62ch; line-height: 1.5;
}}
.fr-head-meta {{ display: flex; gap: 1.6rem; flex-shrink: 0; padding-top: 0.2rem; }}
.fr-meta-item {{ text-align: right; }}
.fr-meta-label {{
  font-size: 0.62rem; color: var(--fr-text-muted);
  text-transform: uppercase; letter-spacing: 0.8px;
}}
.fr-meta-value {{
  font-size: 0.93rem; font-weight: 620; color: var(--fr-text); margin-top: 3px;
  font-variant-numeric: tabular-nums;
}}

/* ---------- status pill ---------- */
.fr-status {{
  display: inline-flex; align-items: center; gap: 0.44rem;
  font-size: 0.71rem; font-weight: 560; letter-spacing: 0.2px;
  padding: 0.24rem 0.62rem; border-radius: 999px;
  background: {p['surface']}; border: 1px solid var(--fr-border);
  color: var(--fr-text-secondary);
}}
.fr-status-dot {{
  width: 6px; height: 6px; border-radius: 50%; flex: 0 0 6px;
  background: var(--fr-positive);
}}
.fr-status-dot.info {{ background: var(--fr-info); }}
.fr-status-dot.muted {{ background: var(--fr-neutral); }}

/* ---------- section header ---------- */
.fr-section {{ margin: 1.9rem 0 0.95rem 0; }}
.fr-section-title {{
  font-size: 1.06rem; font-weight: 630; color: var(--fr-text);
  letter-spacing: -0.25px; margin: 0; display: flex; align-items: center; gap: 0.5rem;
}}
.fr-section-desc {{
  font-size: 0.815rem; color: var(--fr-text-muted);
  margin-top: 0.26rem; max-width: 84ch; line-height: 1.5;
}}
.fr-section-rule {{
  height: 1px; background: var(--fr-border); margin-top: 0.75rem;
}}

/* ---------- KPI card ---------- */
.fr-kpi {{
  background: var(--fr-surface); border: 1px solid var(--fr-border);
  border-radius: var(--fr-radius); padding: 0.85rem 0.95rem 0.8rem 0.95rem;
  height: 100%; position: relative; overflow: hidden;
  transition: border-color 140ms ease;
}}
.fr-kpi:hover {{ border-color: var(--fr-border-strong); }}
.fr-kpi::before {{
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 2px;
  background: var(--fr-accent-bar, transparent);
}}
.fr-kpi-head {{
  display: flex; align-items: center; gap: 0.4rem; margin-bottom: 0.5rem;
}}
.fr-kpi-icon {{ font-size: 0.78rem; opacity: 0.85; line-height: 1; }}
.fr-kpi-label {{
  font-size: 0.645rem; font-weight: 620; color: var(--fr-text-muted);
  text-transform: uppercase; letter-spacing: 0.85px;
}}
.fr-kpi-value {{
  font-size: 1.56rem; font-weight: 660; color: var(--fr-text);
  line-height: 1.05; letter-spacing: -0.7px;
  font-variant-numeric: tabular-nums;
}}
.fr-kpi-delta {{
  font-size: 0.735rem; font-weight: 560; margin-top: 0.34rem;
  font-variant-numeric: tabular-nums;
}}
.fr-kpi-delta.up {{ color: var(--fr-risk); }}
.fr-kpi-delta.down {{ color: var(--fr-positive); }}
.fr-kpi-delta.up-good {{ color: var(--fr-positive); }}
.fr-kpi-delta.down-bad {{ color: var(--fr-risk); }}
.fr-kpi-delta.flat {{ color: var(--fr-text-muted); }}
.fr-kpi-note {{
  font-size: 0.715rem; color: var(--fr-text-muted);
  margin-top: 0.34rem; line-height: 1.4;
}}

/* ---------- generic panel ---------- */
.fr-panel {{
  background: var(--fr-surface); border: 1px solid var(--fr-border);
  border-radius: var(--fr-radius); padding: 1.05rem 1.15rem;
}}
.fr-panel.tight {{ padding: 0.8rem 0.9rem; }}

/* ---------- insight card ---------- */
.fr-insight {{
  background: var(--fr-surface); border: 1px solid var(--fr-border);
  border-left: 2px solid var(--fr-insight-accent, var(--fr-info));
  border-radius: 8px; padding: 0.82rem 0.95rem; margin-bottom: 0.62rem;
}}
.fr-insight-head {{
  display: flex; align-items: center; justify-content: space-between;
  gap: 0.7rem; margin-bottom: 0.4rem;
}}
.fr-insight-title {{
  font-size: 0.875rem; font-weight: 620; color: var(--fr-text); line-height: 1.3;
}}
.fr-insight-body {{
  font-size: 0.805rem; color: var(--fr-text-secondary); line-height: 1.55;
}}
.fr-insight-metric {{
  font-family: {MONO_STACK}; font-size: 0.775rem; font-weight: 600;
  color: var(--fr-text); background: {p['bg_alt']};
  border: 1px solid var(--fr-border); border-radius: 5px;
  padding: 0.1rem 0.4rem; display: inline-block; margin-top: 0.1rem;
  font-variant-numeric: tabular-nums;
}}
.fr-insight-why {{
  font-size: 0.765rem; color: var(--fr-text-muted);
  margin-top: 0.5rem; padding-top: 0.5rem;
  border-top: 1px dashed var(--fr-border); line-height: 1.5;
}}
.fr-insight-why b {{ color: var(--fr-text-secondary); font-weight: 600; }}

/* ---------- badge ---------- */
.fr-badge {{
  display: inline-flex; align-items: center; gap: 0.3rem;
  font-size: 0.635rem; font-weight: 650; text-transform: uppercase;
  letter-spacing: 0.7px; padding: 0.15rem 0.46rem; border-radius: 5px;
  border: 1px solid var(--fr-badge-border, var(--fr-border));
  background: var(--fr-badge-bg, {p['surface_raised']});
  color: var(--fr-badge-fg, var(--fr-text-secondary));
  white-space: nowrap;
}}

/* ---------- signal card ---------- */
.fr-signal {{
  background: var(--fr-surface); border: 1px solid var(--fr-border);
  border-radius: var(--fr-radius); padding: 0.9rem 1rem; height: 100%;
  position: relative;
}}
.fr-signal-top {{
  display: flex; align-items: center; justify-content: space-between;
  gap: 0.6rem; margin-bottom: 0.55rem;
}}
.fr-signal-name {{
  font-size: 0.665rem; font-weight: 660; text-transform: uppercase;
  letter-spacing: 0.85px; color: var(--fr-text-muted);
}}
.fr-signal-value {{
  font-size: 1.32rem; font-weight: 660; color: var(--fr-text);
  line-height: 1.1; letter-spacing: -0.5px; font-variant-numeric: tabular-nums;
}}
.fr-signal-desc {{
  font-size: 0.755rem; color: var(--fr-text-muted);
  margin-top: 0.4rem; line-height: 1.45;
}}
.fr-signal-bar {{
  height: 3px; border-radius: 2px; margin-top: 0.7rem;
  background: {p['bg_alt']}; overflow: hidden;
}}
.fr-signal-bar > span {{ display: block; height: 100%; border-radius: 2px; }}

/* ---------- action chain ---------- */
.fr-chain {{
  background: var(--fr-surface); border: 1px solid var(--fr-border);
  border-radius: var(--fr-radius); padding: 1rem 1.1rem; margin-bottom: 0.7rem;
}}
.fr-chain-step {{ display: flex; gap: 0.8rem; padding: 0.4rem 0; }}
.fr-chain-tag {{
  flex: 0 0 82px; font-size: 0.625rem; font-weight: 680;
  text-transform: uppercase; letter-spacing: 0.9px; padding-top: 0.14rem;
}}
.fr-chain-text {{
  font-size: 0.81rem; color: var(--fr-text-secondary); line-height: 1.55;
}}
.fr-chain-sep {{ height: 1px; background: var(--fr-border); margin: 0.2rem 0; }}

/* ---------- empty / notice ---------- */
.fr-notice {{
  background: var(--fr-surface); border: 1px solid var(--fr-border);
  border-radius: var(--fr-radius); padding: 1.7rem 1.4rem; text-align: center;
}}
.fr-notice-icon {{ font-size: 1.5rem; opacity: 0.55; margin-bottom: 0.5rem; }}
.fr-notice-title {{
  font-size: 0.95rem; font-weight: 620; color: var(--fr-text); margin-bottom: 0.32rem;
}}
.fr-notice-body {{
  font-size: 0.81rem; color: var(--fr-text-muted);
  line-height: 1.55; max-width: 58ch; margin: 0 auto;
}}
.fr-notice.attention {{ border-color: {p['attention_dim']}; }}
.fr-notice.risk {{ border-color: {p['risk_dim']}; }}

/* ---------- pipeline diagram ---------- */
.fr-pipe {{ display: flex; flex-direction: column; gap: 0; }}
.fr-pipe-stage {{
  display: flex; gap: 0.95rem; align-items: flex-start;
  padding: 0.62rem 0.9rem; background: var(--fr-surface);
  border: 1px solid var(--fr-border); border-radius: 8px;
}}
.fr-pipe-num {{
  flex: 0 0 24px; width: 24px; height: 24px; border-radius: 6px;
  background: {p['surface_raised']}; border: 1px solid var(--fr-border-strong);
  color: var(--fr-info); font-size: 0.7rem; font-weight: 680;
  display: flex; align-items: center; justify-content: center;
  font-variant-numeric: tabular-nums;
}}
.fr-pipe-name {{ font-size: 0.855rem; font-weight: 620; color: var(--fr-text); }}
.fr-pipe-desc {{
  font-size: 0.775rem; color: var(--fr-text-muted); margin-top: 0.2rem; line-height: 1.5;
}}
.fr-pipe-arrow {{
  color: {p['text_faint']}; font-size: 0.85rem; text-align: center;
  padding: 0.18rem 0; margin-left: 1.55rem;
}}

/* ---------- filter bar ---------- */
.fr-filter-head {{
  display: flex; align-items: center; justify-content: space-between;
  gap: 1rem; margin-bottom: 0.15rem;
}}
.fr-filter-count {{
  font-size: 0.79rem; color: var(--fr-text-secondary);
  font-variant-numeric: tabular-nums;
}}
.fr-filter-count b {{ color: var(--fr-text); font-weight: 640; }}

/* ---------- streamlit widget polish ---------- */
[data-testid="stExpander"] {{
  background: var(--fr-surface); border: 1px solid var(--fr-border);
  border-radius: var(--fr-radius);
}}
[data-testid="stExpander"] summary {{
  font-size: 0.83rem; font-weight: 570; color: var(--fr-text-secondary);
}}
[data-testid="stExpander"] summary:hover {{ color: var(--fr-text); }}

.stTabs [data-baseweb="tab-list"] {{
  gap: 0.15rem; border-bottom: 1px solid var(--fr-border);
  background: transparent;
}}
.stTabs [data-baseweb="tab"] {{
  height: 38px; padding: 0 0.95rem; background: transparent;
  color: var(--fr-text-muted); font-size: 0.845rem; font-weight: 550;
  border-radius: 7px 7px 0 0; border-bottom: 2px solid transparent;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: var(--fr-text-secondary); }}
.stTabs [aria-selected="true"] {{
  color: var(--fr-text) !important; font-weight: 620;
  border-bottom-color: var(--fr-info) !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{ background: transparent; }}

[data-testid="stMetric"] {{
  background: var(--fr-surface); border: 1px solid var(--fr-border);
  border-radius: 8px; padding: 0.7rem 0.85rem;
}}
[data-testid="stMetricLabel"] {{
  font-size: 0.66rem !important; text-transform: uppercase;
  letter-spacing: 0.8px; color: var(--fr-text-muted) !important;
}}
[data-testid="stMetricValue"] {{
  font-size: 1.32rem !important; font-weight: 640 !important; color: var(--fr-text) !important;
}}

div[data-testid="stDataFrame"] {{
  border: 1px solid var(--fr-border); border-radius: 8px; overflow: hidden;
}}

.stSlider [data-baseweb="slider"] {{ padding-top: 0.2rem; }}
[data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p {{
  font-size: 0.735rem !important; color: var(--fr-text-muted) !important;
  font-weight: 550 !important; text-transform: uppercase; letter-spacing: 0.5px;
}}

.stButton > button {{
  border-radius: 7px; font-size: 0.8rem; font-weight: 570;
  border: 1px solid var(--fr-border-strong); background: {p['surface_raised']};
  color: var(--fr-text-secondary); padding: 0.36rem 0.85rem;
}}
.stButton > button:hover {{
  border-color: {p['info_dim']}; color: var(--fr-text); background: {p['surface_hover']};
}}

hr {{ border-color: var(--fr-border); margin: 1.1rem 0; }}

/* Plotly chart container: remove default white flash. */
.js-plotly-plot .plotly .modebar {{ background: transparent !important; }}
[data-testid="stPlotlyChart"] {{
  background: var(--fr-surface); border: 1px solid var(--fr-border);
  border-radius: var(--fr-radius); padding: 0.5rem 0.55rem 0.3rem 0.55rem;
}}

/* Tighter caption styling. */
[data-testid="stCaptionContainer"] p {{
  font-size: 0.735rem; color: var(--fr-text-muted); line-height: 1.5;
}}

/* ---------- responsive ---------- */
@media (max-width: 1500px) {{
  [data-testid="stAppViewContainer"] > .main .block-container {{
    padding-left: 1.5rem; padding-right: 1.5rem;
  }}
  .fr-kpi-value {{ font-size: 1.4rem; }}
}}
@media (max-width: 1200px) {{
  .fr-page-head {{ flex-direction: column; gap: 0.9rem; }}
  .fr-head-meta {{ gap: 1.5rem; }}
  .fr-meta-item {{ text-align: left; }}
  .fr-page-title {{ font-size: 1.42rem; }}
}}
@media (max-width: 900px) {{
  [data-testid="stAppViewContainer"] > .main .block-container {{
    padding-left: 1rem; padding-right: 1rem;
  }}
  .fr-kpi-value {{ font-size: 1.28rem; }}
  .fr-chain-tag {{ flex-basis: 70px; }}
}}
</style>
"""


def inject_theme() -> None:
    """Apply the stylesheet and register the Plotly template.

    Safe to call on every rerun.
    """
    import streamlit as st

    register_plotly_template()
    st.markdown(build_css(), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Colour helpers
# --------------------------------------------------------------------------- #


def severity_colour(severity: str) -> str:
    """Colour for a severity label, defaulting to neutral."""
    return SEVERITY_COLOURS.get(str(severity).lower(), PALETTE["neutral"])


def finding_colour(finding_type: str) -> str:
    """Colour for a finding type, defaulting to info."""
    return FINDING_COLOURS.get(str(finding_type).lower(), PALETTE["info"])


def with_alpha(hex_colour: str, alpha: float) -> str:
    """Convert a hex colour to an ``rgba()`` string.

    Args:
        hex_colour: Colour as ``#rrggbb``.
        alpha: Opacity between 0 and 1.

    Returns:
        An ``rgba(...)`` string; the input unchanged if it is not parseable.
    """
    value = hex_colour.lstrip("#")
    if len(value) != 6:
        return hex_colour
    try:
        r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return hex_colour
    return f"rgba({r}, {g}, {b}, {max(0.0, min(1.0, alpha)):.3f})"
