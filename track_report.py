"""
Track performance of tickers from a previous infravaloradas report.

Usage:
  python track_report.py infravaloradas/infravaloradas_2026-09-09.txt
  python track_report.py infravaloradas/infravaloradas_2026-09-09.txt --save
"""

from __future__ import annotations

import argparse
import os
import re
import time
from datetime import datetime
from pathlib import Path

import yfinance as yf
from tabulate import tabulate

# Matches summary rows like:
# HBX.MC     | $7.24      | $37.04       | 80.45%       | 3/7   | ...
ROW_RE = re.compile(
    r"^([A-Z0-9.\-]+)\s*\|\s*\$?\s*([\d.,]+)\s*\|\s*\$?\s*([\d.,]+)\s*\|\s*([\d.,]+)\s*%",
    re.IGNORECASE,
)
DATE_IN_HEADER_RE = re.compile(r"INFRAVALORADAS\s*-\s*(\d{4}-\d{2}-\d{2})", re.IGNORECASE)
DATE_IN_FILENAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def parse_number(raw: str) -> float:
    """Parse numbers that may use comma or dot as decimal separator."""
    cleaned = raw.strip().replace("$", "").replace("%", "").replace(" ", "")
    if "," in cleaned and "." in cleaned:
        # Assume US format: 1,234.56
        cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")
    return float(cleaned)


def extract_report_date(path: Path, text: str) -> datetime:
    header_match = DATE_IN_HEADER_RE.search(text)
    if header_match:
        return datetime.strptime(header_match.group(1), "%Y-%m-%d")

    file_match = DATE_IN_FILENAME_RE.search(path.name)
    if file_match:
        return datetime.strptime(file_match.group(1), "%Y-%m-%d")

    raise ValueError(f"No se pudo determinar la fecha del informe: {path}")


def parse_report(path: Path) -> tuple[datetime, list[dict]]:
    """Parse the summary table of an infravaloradas report."""
    text = path.read_text(encoding="utf-8")
    report_date = extract_report_date(path, text)

    entries: list[dict] = []
    in_summary = False

    for line in text.splitlines():
        if "DETALLE EXTENDIDO" in line.upper():
            break
        if "Ticker" in line and "Precio" in line:
            in_summary = True
            continue
        if not in_summary:
            continue
        if line.startswith("---") or not line.strip():
            continue

        match = ROW_RE.match(line.strip())
        if not match:
            continue

        ticker, price_raw, iv_raw, mos_raw = match.groups()
        entries.append(
            {
                "ticker": ticker.upper(),
                "report_price": parse_number(price_raw),
                "intrinsic_value": parse_number(iv_raw),
                "mos": parse_number(mos_raw) / 100.0,
            }
        )

    if not entries:
        raise ValueError(f"No se encontraron tickers en el resumen de: {path}")

    return report_date, entries


def fetch_current_price(ticker: str) -> float | None:
    """Fetch latest available price for a ticker via yfinance."""
    for attempt in range(3):
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            for key in ("currentPrice", "regularMarketPrice", "previousClose", "lastPrice"):
                value = info.get(key)
                if isinstance(value, (int, float)) and value > 0:
                    return float(value)

            hist = t.history(period="5d")
            if hist is not None and not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def format_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2%}"


def color_pct(value: float | None) -> str:
    text = format_pct(value)
    if value is None:
        return text
    if value > 0:
        return f"{Colors.GREEN}{text}{Colors.RESET}"
    if value < 0:
        return f"{Colors.RED}{text}{Colors.RESET}"
    return text


def track_entries(entries: list[dict]) -> list[dict]:
    results = []
    total = len(entries)

    for i, entry in enumerate(entries, start=1):
        ticker = entry["ticker"]
        print(f"[{i}/{total}] Consultando {ticker}...")
        current = fetch_current_price(ticker)
        report_price = entry["report_price"]

        ret = None
        remaining_mos = None
        if current is not None and report_price > 0:
            ret = (current - report_price) / report_price
            iv = entry["intrinsic_value"]
            if iv > 0:
                remaining_mos = (iv - current) / iv

        results.append(
            {
                **entry,
                "current_price": current,
                "return": ret,
                "remaining_mos": remaining_mos,
            }
        )
        time.sleep(0.4)

    return results


def print_results(report_date: datetime, report_path: Path, results: list[dict]) -> None:
    days = (datetime.now() - report_date).days
    print()
    print("=" * 90)
    print(f"SEGUIMIENTO DE INFORME: {report_path.name}")
    print(f"Fecha informe: {report_date.strftime('%Y-%m-%d')}  |  Días transcurridos: {days}")
    print("=" * 90)

    rows = []
    for r in sorted(
        results,
        key=lambda x: (x["return"] is None, -(x["return"] or 0)),
    ):
        rows.append(
            [
                r["ticker"],
                f"${r['report_price']:.2f}",
                f"${r['current_price']:.2f}" if r["current_price"] is not None else "N/A",
                color_pct(r["return"]),
                f"{r['mos']:.1%}",
                format_pct(r["remaining_mos"]) if r["remaining_mos"] is not None else "N/A",
            ]
        )

    print(
        tabulate(
            rows,
            headers=["Ticker", "Precio informe", "Precio hoy", "Retorno", "MOS informe", "MOS restante*"],
            tablefmt="simple",
        )
    )
    print("\n* MOS restante usa el V.I. Normal del informe original (no revalorizado).")

    valid = [r["return"] for r in results if r["return"] is not None]
    failed = [r["ticker"] for r in results if r["return"] is None]

    if valid:
        avg = sum(valid) / len(valid)
        winners = sum(1 for v in valid if v > 0)
        losers = sum(1 for v in valid if v < 0)
        best = max(results, key=lambda x: x["return"] if x["return"] is not None else float("-inf"))
        worst = min(results, key=lambda x: x["return"] if x["return"] is not None else float("inf"))

        print()
        print("-" * 90)
        print(f"Empresas con dato: {len(valid)}/{len(results)}")
        print(f"Retorno medio (igual ponderado): {color_pct(avg)}")
        print(f"Ganadoras / Perdedoras / Planas: {winners} / {losers} / {len(valid) - winners - losers}")
        print(f"Mejor:  {best['ticker']} ({color_pct(best['return'])})")
        print(f"Peor:   {worst['ticker']} ({color_pct(worst['return'])})")
        print("-" * 90)

    if failed:
        print(f"{Colors.YELLOW}Sin precio actual: {', '.join(failed)}{Colors.RESET}")


def save_results(report_date: datetime, report_path: Path, results: list[dict], out_path: Path) -> None:
    days = (datetime.now() - report_date).days
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [
        "=" * 90,
        f"SEGUIMIENTO DE INFORME: {report_path.name}",
        f"Fecha informe: {report_date.strftime('%Y-%m-%d')}  |  Fecha tracking: {today}  |  Días: {days}",
        "=" * 90,
        "",
        f"{'Ticker':<10} | {'P. informe':>10} | {'P. hoy':>10} | {'Retorno':>10} | {'MOS orig':>10} | {'MOS rest':>10}",
        "-" * 90,
    ]

    for r in sorted(results, key=lambda x: (x["return"] is None, -(x["return"] or 0))):
        cur = f"${r['current_price']:.2f}" if r["current_price"] is not None else "N/A"
        ret = format_pct(r["return"])
        mos_rest = format_pct(r["remaining_mos"]) if r["remaining_mos"] is not None else "N/A"
        lines.append(
            f"{r['ticker']:<10} | ${r['report_price']:>9.2f} | {cur:>10} | {ret:>10} | "
            f"{r['mos']:>9.1%} | {mos_rest:>10}"
        )

    valid = [r["return"] for r in results if r["return"] is not None]
    lines.append("")
    if valid:
        avg = sum(valid) / len(valid)
        winners = sum(1 for v in valid if v > 0)
        losers = sum(1 for v in valid if v < 0)
        lines.append(f"Retorno medio (igual ponderado): {format_pct(avg)}")
        lines.append(f"Ganadoras / Perdedoras: {winners} / {losers}")
        lines.append(f"Cobertura: {len(valid)}/{len(results)}")

    lines.append("")
    lines.append("* MOS restante usa el V.I. Normal del informe original (no revalorizado).")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{Colors.GREEN}Informe guardado en: {out_path}{Colors.RESET}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mide el rendimiento de las empresas de un informe infravaloradas desde su fecha."
    )
    parser.add_argument(
        "report",
        help="Ruta al informe (ej. infravaloradas/infravaloradas_2026-09-09.txt)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Guardar el resultado en tracking/tracking_YYYY-MM-DD_from_ORIGDATE.txt",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Ruta de salida personalizada (implica --save)",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        # Allow passing just the date or filename from project root
        alt = Path("infravaloradas") / report_path.name
        if alt.exists():
            report_path = alt
        else:
            raise SystemExit(f"No existe el archivo: {args.report}")

    report_date, entries = parse_report(report_path)
    print(
        f"Informe del {report_date.strftime('%Y-%m-%d')}: "
        f"{len(entries)} tickers encontrados. Consultando precios actuales..."
    )

    results = track_entries(entries)
    print_results(report_date, report_path, results)

    if args.save or args.out:
        if args.out:
            out_path = Path(args.out)
        else:
            today = datetime.now().strftime("%Y-%m-%d")
            out_path = Path("tracking") / (
                f"tracking_{today}_from_{report_date.strftime('%Y-%m-%d')}.txt"
            )
        save_results(report_date, report_path, results, out_path)


if __name__ == "__main__":
    main()
