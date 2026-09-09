"""
Peer / industry-relative checks for value investing decisions.

Builds industry medians from the scanned universe (no extra API calls on batch
runs) and compares each company on valuation, leverage and profitability.
"""

from __future__ import annotations

import json
import os
from statistics import median
from typing import Any

# Minimum peers with usable data before industry medians are trusted
PEER_MIN_SAMPLE = 5

# Relative quality thresholds (MarketInOut-style, adapted to our metrics)
P_FCF_VS_INDUSTRY_MAX = 0.80      # cheaper than industry median
DEBT_EQUITY_VS_INDUSTRY_MAX = 0.80
MARGIN_VS_INDUSTRY_MIN = 1.20     # more profitable than industry median

BENCHMARKS_PATH = os.path.join("data", "industry_benchmarks.json")


def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    try:
        return float(num) / float(den)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _series_avg_ratio(numerators, denominators, max_years: int = 5) -> float | None:
    """Average of year-by-year ratios for up to max_years columns."""
    try:
        ratios = []
        n = min(len(numerators), len(denominators), max_years)
        for i in range(n):
            den = denominators.iloc[i]
            num = numerators.iloc[i]
            if den and den != 0 and num is not None:
                ratios.append(float(num) / float(den))
        if not ratios:
            return None
        return sum(ratios) / len(ratios)
    except Exception:
        return None


def extract_peer_metrics(
    info: dict,
    financials,
    cashflow,
    fcf: float | None,
    debt_to_equity: float | None,
    operating_margin: float | None,
    roe: float | None,
) -> dict[str, Any]:
    """
    Metrics used for industry comparison. Prefer multi-year averages when
    statements have enough history (Yahoo typically gives ~4 years).
    """
    industry = info.get("industry") or "Unknown"
    sector = info.get("sector") or "Unknown"
    market_cap = info.get("marketCap")

    # Price / Free Cash Flow (absolute FCF, 3Y avg when available)
    price_to_fcf = _safe_div(market_cap, fcf) if fcf and fcf > 0 else None

    # Operating cash flow multiple as fallback valuation check
    ocf = info.get("operatingCashflow")
    price_to_ocf = _safe_div(market_cap, ocf) if ocf and ocf > 0 else None

    profit_margin = info.get("profitMargins")

    op_margin_avg = None
    try:
        if financials is not None and not financials.empty:
            if "Operating Income" in financials.index and "Total Revenue" in financials.index:
                op_margin_avg = _series_avg_ratio(
                    financials.loc["Operating Income"],
                    financials.loc["Total Revenue"],
                    max_years=5,
                )
    except Exception:
        op_margin_avg = None

    # Prefer multi-year op margin for peer compare; fall back to TTM
    margin_for_peers = op_margin_avg if op_margin_avg is not None else operating_margin

    return {
        "industry": industry,
        "sector": sector,
        "market_cap": market_cap,
        "Price/FCF": price_to_fcf,
        "Price/OCF": price_to_ocf,
        "Debt-to-Equity": debt_to_equity,
        "Operating Margin": margin_for_peers,
        "Profit Margin": profit_margin,
        "ROE": roe,
        "Operating Margin (TTM)": operating_margin,
        "Operating Margin (multi-year avg)": op_margin_avg,
    }


def extract_peer_metrics_with_balance(
    info: dict,
    financials,
    balance_sheet,
    cashflow,
    fcf: float | None,
    debt_to_equity: float | None,
    operating_margin: float | None,
    roe: float | None,
) -> dict[str, Any]:
    """Same as extract_peer_metrics, plus multi-year ROE from statements."""
    metrics = extract_peer_metrics(
        info, financials, cashflow, fcf, debt_to_equity, operating_margin, roe
    )
    roe_avg = None
    try:
        if (
            financials is not None
            and balance_sheet is not None
            and not financials.empty
            and not balance_sheet.empty
            and "Net Income" in financials.index
        ):
            equity_key = next(
                (
                    k
                    for k in (
                        "Stockholders Equity",
                        "Total Stockholder Equity",
                        "Common Stock Equity",
                    )
                    if k in balance_sheet.index
                ),
                None,
            )
            if equity_key:
                roe_avg = _series_avg_ratio(
                    financials.loc["Net Income"],
                    balance_sheet.loc[equity_key],
                    max_years=5,
                )
    except Exception:
        roe_avg = None

    metrics["ROE (multi-year avg)"] = roe_avg
    if roe_avg is not None:
        metrics["ROE"] = roe_avg
    return metrics


def _median_of(values: list[float]) -> float | None:
    clean = [v for v in values if isinstance(v, (int, float)) and v == v]
    if len(clean) < PEER_MIN_SAMPLE:
        return None
    return float(median(clean))


def build_industry_benchmarks(analyses: list[dict]) -> dict[str, Any]:
    """
    Industry medians from a full scan. Only industries with enough samples
    get usable medians.
    """
    by_industry: dict[str, dict[str, list[float]]] = {}

    for analysis in analyses:
        peer_m = analysis.get("peer_metrics") or {}
        industry = peer_m.get("industry") or "Unknown"
        if industry == "Unknown":
            continue
        bucket = by_industry.setdefault(
            industry,
            {
                "Price/FCF": [],
                "Debt-to-Equity": [],
                "Operating Margin": [],
                "Profit Margin": [],
                "ROE": [],
            },
        )
        for key in bucket:
            val = peer_m.get(key)
            if isinstance(val, (int, float)) and val == val:
                # Skip non-sensical negatives for multiples / D/E
                if key in ("Price/FCF", "Debt-to-Equity") and val <= 0:
                    continue
                bucket[key].append(float(val))

    industries: dict[str, Any] = {}
    for industry, series in by_industry.items():
        sample_sizes = {k: len(v) for k, v in series.items()}
        industries[industry] = {
            "sample_size": max(sample_sizes.values()) if sample_sizes else 0,
            "sample_sizes": sample_sizes,
            "medians": {
                "Price/FCF": _median_of(series["Price/FCF"]),
                "Debt-to-Equity": _median_of(series["Debt-to-Equity"]),
                "Operating Margin": _median_of(series["Operating Margin"]),
                "Profit Margin": _median_of(series["Profit Margin"]),
                "ROE": _median_of(series["ROE"]),
            },
        }

    return {
        "version": 1,
        "peer_min_sample": PEER_MIN_SAMPLE,
        "industries": industries,
    }


def save_benchmarks(benchmarks: dict, path: str = BENCHMARKS_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)


def load_benchmarks(path: str = BENCHMARKS_PATH) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _pct_of_industry(company: float | None, industry_med: float | None) -> float | None:
    """company / industry_median (1.0 = in line with industry)."""
    if company is None or industry_med is None or industry_med == 0:
        return None
    return float(company) / float(industry_med)


def compare_to_peers(analysis: dict, benchmarks: dict | None) -> dict[str, Any]:
    """
    Relative standing vs industry medians.

    Edges (good for value):
      - Price/FCF vs industry < 80%
      - Debt/Equity vs industry < 80%
      - Operating (or profit) margin vs industry > 120%

    Robustness rule: if peers are usable and the name is worse on EVERY
    available relative check, treat as a relative trap (exclude from gems).
    Insufficient peer data does not block.
    """
    peer_m = analysis.get("peer_metrics") or {}
    industry = peer_m.get("industry") or "Unknown"

    empty = {
        "industry": industry,
        "status": "unavailable",
        "usable": False,
        "edge_score": 0,
        "max_edges": 3,
        "relative": {},
        "edges": {},
        "is_relative_trap": False,
        "summary": "Sin datos de peers / industria.",
    }

    if not benchmarks or industry == "Unknown":
        return empty

    ind = (benchmarks.get("industries") or {}).get(industry)
    if not ind:
        return {
            **empty,
            "summary": f"Sin benchmark para industria '{industry}'. Ejecuta un scan completo primero.",
        }

    medians = ind.get("medians") or {}
    sample_sizes = ind.get("sample_sizes") or {}

    p_fcf = peer_m.get("Price/FCF")
    d_e = peer_m.get("Debt-to-Equity")
    op_m = peer_m.get("Operating Margin")
    # Prefer operating margin; fall back to net profit margin vs its industry median
    margin_company = op_m if op_m is not None else peer_m.get("Profit Margin")
    margin_med_key = "Operating Margin" if op_m is not None else "Profit Margin"
    margin_med = medians.get(margin_med_key)

    rel_p_fcf = _pct_of_industry(p_fcf, medians.get("Price/FCF"))
    rel_de = _pct_of_industry(d_e, medians.get("Debt-to-Equity"))
    rel_margin = _pct_of_industry(margin_company, margin_med)

    # Only count a check when both company and industry median exist and sample OK
    checks: list[tuple[str, bool | None]] = []

    def sample_ok(metric: str) -> bool:
        return sample_sizes.get(metric, 0) >= PEER_MIN_SAMPLE and medians.get(metric) is not None

    edges = {
        "cheap_vs_industry": False,
        "less_levered_vs_industry": False,
        "higher_margin_vs_industry": False,
    }

    if sample_ok("Price/FCF") and rel_p_fcf is not None:
        cheap = rel_p_fcf < P_FCF_VS_INDUSTRY_MAX
        edges["cheap_vs_industry"] = cheap
        checks.append(("valuation", cheap))

    if sample_ok("Debt-to-Equity") and rel_de is not None:
        less_debt = rel_de < DEBT_EQUITY_VS_INDUSTRY_MAX
        edges["less_levered_vs_industry"] = less_debt
        checks.append(("leverage", less_debt))

    margin_sample_key = margin_med_key
    if sample_ok(margin_sample_key) and rel_margin is not None:
        better_margin = rel_margin > MARGIN_VS_INDUSTRY_MIN
        edges["higher_margin_vs_industry"] = better_margin
        checks.append(("margin", better_margin))

    usable = len(checks) >= 2  # need at least two relative dimensions
    edge_score = sum(1 for _, ok in checks if ok)

    # Relative trap: usable peers and worse on every available check
    is_trap = usable and len(checks) > 0 and all(ok is False for _, ok in checks)

    if not usable:
        status = "insufficient_peers"
        summary = (
            f"Peers insuficientes en '{industry}' "
            f"(n≈{ind.get('sample_size', 0)}; mínimo {PEER_MIN_SAMPLE} por métrica)."
        )
    elif is_trap:
        status = "relative_trap"
        summary = (
            f"Peor que la mediana de '{industry}' en valoración, deuda y márgenes "
            f"relativos disponibles (posible trampa de valor)."
        )
    elif edge_score >= 2:
        status = "strong_peer_edge"
        summary = f"Ventaja relativa clara vs '{industry}' ({edge_score}/{len(checks)} checks)."
    elif edge_score == 1:
        status = "mixed_peer_edge"
        summary = f"Ventaja mixta vs '{industry}' ({edge_score}/{len(checks)} checks)."
    else:
        status = "no_peer_edge"
        summary = f"Sin ventaja clara vs '{industry}', pero no dominado en todos los checks."

    return {
        "industry": industry,
        "sector": peer_m.get("sector"),
        "status": status,
        "usable": usable,
        "edge_score": edge_score,
        "max_edges": len(checks),
        "industry_sample_size": ind.get("sample_size"),
        "relative": {
            "Price/FCF_vs_industry": rel_p_fcf,
            "Debt/Equity_vs_industry": rel_de,
            "Margin_vs_industry": rel_margin,
            "industry_medians": {
                "Price/FCF": medians.get("Price/FCF"),
                "Debt-to-Equity": medians.get("Debt-to-Equity"),
                "Operating Margin": medians.get("Operating Margin"),
                "Profit Margin": medians.get("Profit Margin"),
            },
        },
        "edges": edges,
        "is_relative_trap": is_trap,
        "summary": summary,
    }


def passes_peer_robustness(peer_comparison: dict | None) -> bool:
    """Do not block on missing peers; block only clear relative traps."""
    if not peer_comparison:
        return True
    return not peer_comparison.get("is_relative_trap", False)


def format_peer_section(peer_comparison: dict | None) -> list[str]:
    """Human-readable lines for terminal / report files."""
    if not peer_comparison:
        return ["Peers: N/A"]

    lines = [
        f"Industry: {peer_comparison.get('industry', 'N/A')}",
        f"Peer status: {peer_comparison.get('status', 'N/A')} "
        f"(edge {peer_comparison.get('edge_score', 0)}/{peer_comparison.get('max_edges', 0)})",
        f"Summary: {peer_comparison.get('summary', '')}",
    ]
    rel = peer_comparison.get("relative") or {}
    for label, key in (
        ("P/FCF vs industry", "Price/FCF_vs_industry"),
        ("D/E vs industry", "Debt/Equity_vs_industry"),
        ("Margin vs industry", "Margin_vs_industry"),
    ):
        val = rel.get(key)
        if isinstance(val, (int, float)):
            lines.append(f"{label}: {val:.0%} of industry median")
        else:
            lines.append(f"{label}: N/A")
    return lines
