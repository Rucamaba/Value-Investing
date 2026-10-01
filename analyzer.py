"""
This script analyzes tickers to identify buying opportunities based on the Value Investing strategy.
"""

import yfinance as yf
import pandas as pd
import os
import numpy as np
import argparse
from markets import get_tickers_from_csv
from peers import (
    build_industry_benchmarks,
    compare_to_peers,
    extract_peer_metrics_with_balance,
    format_peer_section,
    is_financial_firm,
    load_benchmarks,
    passes_peer_robustness,
    save_benchmarks,
)
import time
from datetime import datetime
import concurrent.futures
import random

# --- CONFIGURATION ---
DISCOUNT_RATE_NORMAL = 0.10  # 10%
DISCOUNT_RATE_PESSIMISTIC = 0.12 # 12%
DISCOUNT_RATE_OPTIMISTIC = 0.08 # 8%
DISCOUNT_RATE_ULTRA_PESSIMISTIC = 0.15 # 15%
DISCOUNT_RATE_ULTRA_OPTIMISTIC = 0.05 # 5%
PERPETUAL_GROWTH_RATE = 0.02 # 2%

# Cost of equity for banks / insurers (Gordon / Excess Returns). FCF-DCF
# does not apply: operating cash flow is mostly premiums, claims and float.
KE_FINANCIAL = {
    "Ultra Pessimistic": 0.12,
    "Pessimistic": 0.105,
    "Normal": 0.09,
    "Optimistic": 0.08,
    "Ultra Optimistic": 0.07,
}
G_TERMINAL_FINANCIAL = {
    "Ultra Pessimistic": 0.015,
    "Pessimistic": 0.020,
    "Normal": 0.025,
    "Optimistic": 0.030,
    "Ultra Optimistic": 0.035,
}

SCENARIO_NAMES = (
    "Ultra Pessimistic",
    "Pessimistic",
    "Normal",
    "Optimistic",
    "Ultra Optimistic",
)

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'


def currency_symbol(currency: str | None) -> str:
    mapping = {
        "EUR": "€",
        "USD": "$",
        "GBP": "£",
        "CHF": "CHF ",
        "JPY": "¥",
        "CAD": "C$",
        "AUD": "A$",
        "SEK": "SEK ",
        "NOK": "NOK ",
        "DKK": "DKK ",
        "PLN": "PLN ",
        "HKD": "HK$",
        "CNY": "CN¥",
        "INR": "₹",
    }
    code = (currency or "USD").upper()
    return mapping.get(code, f"{code} ")


def _na_scenarios(reason: str | None = None) -> dict:
    return {sc: (reason or "N/A") for sc in SCENARIO_NAMES}

def get_financial_data(ticker_symbol):
    """
    Downloads and returns financial data for a given ticker with a retry mechanism.
    Updated to let yfinance handle its own internal session (curl_cffi).
    """
    for attempt in range(3): 
        try:
            # 1. Pausa aleatoria ANTES de pedir datos (crucial para 5 hilos)
            # Aumentamos un poco el rango para dar respiro a la API
            time.sleep(random.uniform(1.5, 3.0))
            
            # 2. Creamos el objeto Ticker SIN pasarle session
            ticker = yf.Ticker(ticker_symbol)
            
            # 3. Intentamos obtener los datos
            info = ticker.info
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cashflow = ticker.cashflow
            
            # Validamos que no esté vacío
            if not info or financials.empty or balance_sheet.empty or cashflow.empty:
                # Si falla por datos incompletos, esperamos y reintentamos
                time.sleep(2)
                continue

            # Si llegamos aquí, ordenamos y devolvemos
            for df in [financials, balance_sheet, cashflow]:
                df.sort_index(axis=1, ascending=False, inplace=True)

            return {
                "info": info,
                "financials": financials,
                "balance_sheet": balance_sheet,
                "cashflow": cashflow
            }
            
        except Exception as e:
            # Si el error es de Rate Limit (429), pausa larga
            error_msg = str(e)
            wait_time = 15 if "429" in error_msg or "Too Many Requests" in error_msg else 3
            
            print(f"{Colors.YELLOW}Warning: Failed to get data for {ticker_symbol} on attempt {attempt + 1}. Retrying in {wait_time}s...{Colors.RESET}")
            time.sleep(wait_time)
            
    print(f"{Colors.RED}Fatal: Could not retrieve data for {ticker_symbol} after multiple attempts.{Colors.RESET}")
    return None

def score_moat(info, financials, cashflow, balance_sheet, total_debt, cash, is_financial=False):
    """
    Scores the company's competitive moat based on profitability and consistency.
    Financials use ROE / dividend / EPS stability instead of ROIC / CapEx / interest cover.
    """
    score = 0

    if is_financial:
        try:
            net_income = financials.loc["Net Income"].iloc[0]
            equity = balance_sheet.loc["Stockholders Equity"].iloc[0]
            roe_calc = net_income / equity if equity else 0
            if roe_calc > 0.17:
                score += 2
            elif roe_calc > 0.12:
                score += 1
        except Exception:
            roe_info = info.get("returnOnEquity")
            if isinstance(roe_info, (int, float)):
                if roe_info > 0.17:
                    score += 2
                elif roe_info > 0.12:
                    score += 1

        try:
            ni = financials.loc["Net Income"]
            if len(ni) >= 3 and ni.iloc[0] >= ni.iloc[-1] and ni.iloc[0] > 0:
                score += 2
        except KeyError:
            pass

        try:
            dy = info.get("trailingAnnualDividendYield") or info.get("dividendYield") or 0
            if isinstance(dy, (int, float)):
                dy = dy / 100 if dy > 1 else dy
                if 0.02 <= dy <= 0.08:
                    score += 1
        except Exception:
            pass

        try:
            eps = financials.loc["Basic EPS"]
            if len(eps) >= 3 and eps.iloc[0] > eps.iloc[-1] and eps.iloc[0] > 0:
                score += 1
        except KeyError:
            pass

        try:
            equity_series = balance_sheet.loc["Stockholders Equity"]
            if len(equity_series) >= 2 and equity_series.iloc[0] >= equity_series.iloc[-1]:
                score += 1
        except KeyError:
            pass

        return score

    # --- Non-financial (industrial / consumer) moat ---
    try:
        ebit = financials.loc['EBIT'].iloc[0]
        cap_inv = total_debt + balance_sheet.loc['Stockholders Equity'].iloc[0] - cash
        if cap_inv > 0:
            roic_calc = (ebit * 0.75) / cap_inv
            if roic_calc > 0.15:
                score += 2
    except Exception:
        pass

    try:
        op_income = financials.loc['Operating Income']
        revenue = financials.loc['Total Revenue']
        margins = op_income / revenue
        if len(margins) >= 3 and margins.iloc[0] >= margins.iloc[-1]:
            score += 2
    except KeyError:
        pass

    try:
        cfo = cashflow.loc['Total Cash From Operating Activities'].iloc[0]
        capex = cashflow.loc['Capital Expenditures'].iloc[0]
        if cfo > 0 and (abs(capex) / cfo) < 0.30:
            score += 1
    except (KeyError, IndexError):
        pass

    try:
        eps = financials.loc['Basic EPS']
        if len(eps) >= 3 and eps.iloc[0] > eps.iloc[-1]:
            score += 1
    except KeyError:
        pass

    try:
        ebit = financials.loc['EBIT'].iloc[0]
        interest_exp = abs(financials.loc['Interest Expense'].iloc[0])
        if interest_exp > 0 and (ebit / interest_exp) > 5:
            score += 1
    except Exception:
        pass

    return score

def calculate_intrinsic_value(fcf_current, g_ultra_pessimistic, g_pessimistic, g_normal, g_optimistic, g_ultra_optimistic,
                              shares_outstanding, cash, total_debt, current_price):
    """
    Calculates the intrinsic value using a DCF model for pessimistic, normal, and optimistic scenarios.
    """
    scenarios = {
        "Ultra Pessimistic": {"g": g_ultra_pessimistic, "r": DISCOUNT_RATE_ULTRA_PESSIMISTIC},
        "Pessimistic": {"g": g_pessimistic, "r": DISCOUNT_RATE_PESSIMISTIC},
        "Normal": {"g": g_normal, "r": DISCOUNT_RATE_NORMAL},
        "Optimistic": {"g": g_optimistic, "r": DISCOUNT_RATE_OPTIMISTIC},
        "Ultra Optimistic": {"g": g_ultra_optimistic, "r": DISCOUNT_RATE_ULTRA_OPTIMISTIC},
    }
    
    results = {}

    for scenario_name, params in scenarios.items():
        g = params["g"]
        r = params["r"]

        if r <= PERPETUAL_GROWTH_RATE:
            results[scenario_name] = "Invalid Discount Rate"
            continue

        fcf_projections = []
        last_fcf = fcf_current
        
        for _ in range(10):
            last_fcf *= (1 + g)
            fcf_projections.append(last_fcf)
            
        dcf = [fcf / ((1 + r) ** (i + 1)) for i, fcf in enumerate(fcf_projections)]
        
        fcf_year_10 = fcf_projections[-1]
        terminal_value = fcf_year_10 * (1 + PERPETUAL_GROWTH_RATE) / (r - PERPETUAL_GROWTH_RATE)
        discounted_terminal_value = terminal_value / ((1 + r) ** 10)
        
        enterprise_value = sum(dcf) + discounted_terminal_value
        equity_value = enterprise_value + cash - total_debt
        
        intrinsic_value_per_share = equity_value / shares_outstanding if shares_outstanding else 0
        
        # --- Sanity Checks ---
        if intrinsic_value_per_share < 0:
            intrinsic_value_per_share = 0
        if current_price and intrinsic_value_per_share > current_price * 20:
            results[scenario_name] = "Check Data (Outlier)"
            continue

        results[scenario_name] = intrinsic_value_per_share
        
    return results


def _shareholder_cash_per_share(info, cashflow, shares_outstanding) -> float | None:
    """
    True cash returned to shareholders: dividends + net buybacks, per share.
    Falls back to trailing dividend rate when cash-flow detail is missing.
    """
    dps = info.get("dividendRate") or info.get("trailingAnnualDividendRate")
    buyback_ps = 0.0

    if cashflow is not None and not cashflow.empty and shares_outstanding:
        try:
            repurchase_key = next(
                (
                    k
                    for k in ("Repurchase Of Capital Stock", "Common Stock Repurchased")
                    if k in cashflow.index
                ),
                None,
            )
            issuance_key = next(
                (
                    k
                    for k in ("Issuance Of Capital Stock", "Common Stock Issuance")
                    if k in cashflow.index
                ),
                None,
            )
            repurchase = 0.0
            issuance = 0.0
            if repurchase_key:
                repurchase = abs(float(cashflow.loc[repurchase_key].iloc[:3].mean()))
            if issuance_key:
                issuance = abs(float(cashflow.loc[issuance_key].iloc[:3].mean()))
            net_buyback = max(repurchase - issuance, 0.0)
            buyback_ps = net_buyback / float(shares_outstanding)
        except Exception:
            buyback_ps = 0.0

    if isinstance(dps, (int, float)) and dps > 0:
        return float(dps) + buyback_ps
    if buyback_ps > 0:
        return buyback_ps
    return None


def _book_value_per_share(info, balance_sheet, shares_outstanding) -> float | None:
    bvps = info.get("bookValue")
    if isinstance(bvps, (int, float)) and bvps > 0:
        return float(bvps)
    if not shares_outstanding:
        return None
    try:
        equity_key = next(
            (
                k
                for k in (
                    "Stockholders Equity",
                    "Total Stockholder Equity",
                    "Common Stock Equity",
                    "Tangible Book Value",
                )
                if k in balance_sheet.index
            ),
            None,
        )
        if equity_key:
            equity = float(balance_sheet.loc[equity_key].iloc[0])
            if equity > 0:
                return equity / float(shares_outstanding)
    except Exception:
        pass
    return None


def _normalized_eps(info, financials) -> float | None:
    """Prefer trailing EPS; else multi-year average of Basic EPS."""
    eps = info.get("trailingEps") or info.get("epsTrailingTwelveMonths")
    if isinstance(eps, (int, float)) and eps > 0:
        return float(eps)
    try:
        for key in ("Basic EPS", "Diluted EPS"):
            if key in financials.index:
                series = financials.loc[key].dropna().iloc[:3]
                if not series.empty and series.mean() > 0:
                    return float(series.mean())
    except Exception:
        pass
    return None


def calculate_financial_intrinsic_value(
    dps_effective: float | None,
    bvps: float | None,
    roe: float | None,
    eps: float | None,
    current_price: float | None,
):
    """
    Banks / insurers: blend Gordon DDM (dividends + buybacks) and Excess Returns
    (justified P/B), with a normalized P/E cross-check.
    """
    results = {}
    details = {}
    pe_multiples = {
        "Ultra Pessimistic": 8.0,
        "Pessimistic": 10.0,
        "Normal": 12.0,
        "Optimistic": 13.5,
        "Ultra Optimistic": 15.0,
    }

    for scenario in SCENARIO_NAMES:
        ke = KE_FINANCIAL[scenario]
        g = G_TERMINAL_FINANCIAL[scenario]
        if ke <= g:
            results[scenario] = "Invalid Discount Rate"
            continue

        components = []
        ddm_iv = er_iv = pe_iv = None

        if dps_effective and dps_effective > 0:
            ddm_iv = dps_effective * (1 + g) / (ke - g)
            components.append(ddm_iv)

        if bvps and bvps > 0 and isinstance(roe, (int, float)) and roe > g:
            er_iv = bvps * (roe - g) / (ke - g)
            components.append(er_iv)

        if eps and eps > 0 and components:
            pe_iv = eps * pe_multiples[scenario]
            components.append(pe_iv)

        if not components:
            results[scenario] = "N/A"
            continue

        iv = sum(components) / len(components)
        if iv < 0:
            iv = 0
        if current_price and iv > current_price * 8:
            results[scenario] = "Check Data (Outlier)"
            continue

        results[scenario] = iv
        details[scenario] = {
            "Ke": ke,
            "g": g,
            "DDM": ddm_iv,
            "ExcessReturns": er_iv,
            "PE_norm": pe_iv,
        }

    return results, details


def analyze_ticker(ticker_symbol):
    data = get_financial_data(ticker_symbol)
    if not data:
        print(f"{Colors.YELLOW}Warning: Could not retrieve sufficient financial data for {ticker_symbol}. Skipping...{Colors.RESET}")
        return None

    info = data["info"]
    financials = data["financials"]
    balance_sheet = data["balance_sheet"]
    cashflow = data["cashflow"]

    sector = info.get("sector")
    industry = info.get("industry")
    financial_firm = is_financial_firm(sector, industry)
    currency = info.get("currency") or info.get("financialCurrency") or "USD"
    cur = currency_symbol(currency)

    # --- Ratios de Valoración ---
    pe_ratio = info.get("trailingPE")
    pb_ratio = info.get("priceToBook")
    ev_to_ebitda = None if financial_firm else info.get("enterpriseToEbitda")

    # --- Solvencia y Salud Financiera ---
    # Debt / Current Ratio / Net Debt are misleading for banks & insurers
    # (regulatory Tier debt, float, technical provisions).
    debt_to_equity = None if financial_firm else info.get("debtToEquity")
    current_ratio = None if financial_firm else info.get("currentRatio")

    cash_options = [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash",
    ]
    cash = 0
    for opt in cash_options:
        if opt in balance_sheet.index:
            cash = balance_sheet.loc[opt].iloc[0]
            break

    total_debt = info.get("totalDebt") or 0

    if not financial_firm and current_ratio is None:
        try:
            current_assets = balance_sheet.loc["Total Current Assets"].iloc[0]
            current_liabilities = balance_sheet.loc["Total Current Liabilities"].iloc[0]
            current_ratio = current_assets / current_liabilities
        except Exception:
            pass

    # --- Rentabilidad y Eficiencia ---
    try:
        net_income = financials.loc["Net Income"].iloc[0]
        equity = balance_sheet.loc["Stockholders Equity"].iloc[0]
        roe = net_income / equity if equity > 0 else None
    except Exception:
        roe = info.get("returnOnEquity")

    roic = None
    if not financial_firm:
        try:
            ebit = financials.loc["EBIT"].iloc[0]
            equity = balance_sheet.loc["Stockholders Equity"].iloc[0]
            invested_capital = total_debt + equity - cash
            roic = (ebit * 0.75) / invested_capital if invested_capital > 0 else None
        except Exception:
            roic = info.get("returnOnInvestedCapital")

    gross_margin = None if financial_firm else info.get("grossMargins")
    operating_margin = None if financial_firm else info.get("operatingMargins")

    # --- Flujo de Caja (informativo; no base de IV en financieras) ---
    try:
        if "Free Cash Flow" in cashflow.index:
            fcf_series = cashflow.loc["Free Cash Flow"].dropna()
        else:
            cfo_key = next(
                (
                    x
                    for x in ("Operating Cash Flow", "Total Cash From Operating Activities")
                    if x in cashflow.index
                ),
                None,
            )
            capex_key = next(
                (
                    x
                    for x in ("Capital Expenditure", "Investing Cash Flow")
                    if x in cashflow.index
                ),
                None,
            )
            if cfo_key and capex_key:
                fcf_series = (cashflow.loc[cfo_key] - abs(cashflow.loc[capex_key])).dropna()
            else:
                fcf_series = pd.Series(dtype=float)

        fcf = fcf_series.iloc[:3].mean() if not fcf_series.empty else None
    except Exception:
        fcf = None

    dividend_yield = info.get("trailingAnnualDividendYield") or info.get("dividendYield")

    # --- Cálculo de Crecimiento (CAGR) — solo para DCF no financiero ---
    try:
        revenues = financials.loc["Total Revenue"].dropna()
        if len(revenues) > 1:
            num_years = len(revenues) - 1
            cagr = (revenues.iloc[0] / revenues.iloc[-1]) ** (1 / num_years) - 1
            if cagr < 0:
                cagr = 0.01
            g_normal = min(cagr * 0.7, 0.15)
        else:
            g_normal = 0.05

        g_ultra_pessimistic = g_normal * 0.3
        g_pessimistic = g_normal * 0.6
        g_optimistic = g_normal * 1.3
        g_ultra_optimistic = g_normal * 1.6
    except Exception:
        g_normal = 0.05
        g_ultra_pessimistic, g_pessimistic = 0.01, 0.03
        g_optimistic, g_ultra_optimistic = 0.07, 0.10

    shares_outstanding = info.get("impliedSharesOutstanding") or info.get("sharesOutstanding")
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")

    # --- Cálculo del Valor Intrínseco ---
    error_reason = None
    valuation_model = "DDM+ExcessReturns" if financial_firm else "DCF"
    financial_iv_details = None
    dps_effective = None
    bvps = None
    eps_norm = None

    if financial_firm:
        dps_effective = _shareholder_cash_per_share(info, cashflow, shares_outstanding)
        bvps = _book_value_per_share(info, balance_sheet, shares_outstanding)
        eps_norm = _normalized_eps(info, financials)

        if not shares_outstanding and not dps_effective and not bvps:
            error_reason = "No Shares/Dividend/Book Data"
            intrinsic_values = _na_scenarios()
        elif not dps_effective and not (bvps and roe):
            error_reason = "No Dividend/ROE Data for Financial Model"
            intrinsic_values = _na_scenarios()
        else:
            intrinsic_values, financial_iv_details = calculate_financial_intrinsic_value(
                dps_effective=dps_effective,
                bvps=bvps,
                roe=roe,
                eps=eps_norm,
                current_price=current_price,
            )
            if isinstance(intrinsic_values.get("Normal"), str):
                error_reason = intrinsic_values.get("Normal")
    else:
        if fcf is None or fcf <= 0:
            error_reason = "Negative/Zero FCF"
        elif not shares_outstanding:
            error_reason = "No Shares Data"

        if error_reason:
            intrinsic_values = _na_scenarios()
        else:
            intrinsic_values = calculate_intrinsic_value(
                fcf_current=float(fcf),
                g_ultra_pessimistic=g_ultra_pessimistic,
                g_pessimistic=g_pessimistic,
                g_normal=g_normal,
                g_optimistic=g_optimistic,
                g_ultra_optimistic=g_ultra_optimistic,
                shares_outstanding=float(shares_outstanding),
                cash=float(cash),
                total_debt=float(total_debt),
                current_price=current_price,
            )
            if isinstance(intrinsic_values.get("Normal"), str):
                error_reason = intrinsic_values.get("Normal")

    margin_of_safety = {}
    if current_price:
        for scenario, iv in intrinsic_values.items():
            if isinstance(iv, (int, float)) and iv > 0:
                margin_of_safety[scenario] = (iv - current_price) / iv
            else:
                margin_of_safety[scenario] = None

    moat_score = score_moat(
        info, financials, cashflow, balance_sheet, total_debt, cash, is_financial=financial_firm
    )

    peer_metrics = extract_peer_metrics_with_balance(
        info=info,
        financials=financials,
        balance_sheet=balance_sheet,
        cashflow=cashflow,
        fcf=None if financial_firm else fcf,
        debt_to_equity=debt_to_equity,
        operating_margin=operating_margin,
        roe=roe,
    )

    valuation = {
        "P/E Ratio": pe_ratio,
        "P/BV Ratio": pb_ratio,
    }
    if financial_firm:
        valuation["EPS (normalized)"] = eps_norm
        valuation["BVPS"] = bvps
    else:
        valuation["EV/EBITDA"] = ev_to_ebitda
        valuation["Price/FCF"] = peer_metrics.get("Price/FCF")

    solvency = {}
    if financial_firm:
        solvency["Note"] = "D/E, Current Ratio y Net Debt no aplican (float / Tier debt / reservas)"
    else:
        solvency = {
            "Debt-to-Equity": debt_to_equity,
            "Current Ratio": current_ratio,
            "Net Debt": total_debt - cash,
        }

    profitability = {
        "ROE": roe,
        "ROE (multi-year avg)": peer_metrics.get("ROE (multi-year avg)"),
    }
    if not financial_firm:
        profitability.update(
            {
                "ROIC": roic,
                "Gross Margin": gross_margin,
                "Operating Margin": operating_margin,
                "Operating Margin (multi-year avg)": peer_metrics.get(
                    "Operating Margin (multi-year avg)"
                ),
            }
        )

    cash_flow = {
        "Dividend Yield": dividend_yield,
    }
    if financial_firm:
        cash_flow["Dividend + Buybacks / share"] = dps_effective
        cash_flow["FCF (informational, not used)"] = fcf
    else:
        cash_flow["Free Cash Flow (3Y Avg)"] = fcf

    return {
        "ticker": ticker_symbol,
        "price": current_price,
        "currency": currency,
        "currency_symbol": cur,
        "is_financial": financial_firm,
        "valuation_model": valuation_model,
        "error_reason": error_reason,
        "moat_score": moat_score,
        "valuation": valuation,
        "solvency": solvency,
        "profitability": profitability,
        "cash_flow": cash_flow,
        "financial_iv_details": financial_iv_details,
        "peer_metrics": peer_metrics,
        "peers": None,
        "intrinsic_value": intrinsic_values,
        "margin_of_safety": margin_of_safety,
        "company_name": info.get("longName"),
        "industry": peer_metrics.get("industry") or industry,
        "sector": peer_metrics.get("sector") or sector,
    }


def is_quality_gem(analysis: dict) -> bool:
    """
    Core gem definition: MOS + moat + red-flag filters + peer robustness.
    Peer data only blocks clear relative traps (worse vs industry on all checks).
    Financials skip industrial leverage / operating-margin red flags.
    """
    mos_normal = analysis.get("margin_of_safety", {}).get("Normal")
    moat = analysis.get("moat_score")
    solvency = analysis.get("solvency", {})
    profitability = analysis.get("profitability", {})
    is_financial = analysis.get("is_financial", False)
    debt_to_equity = solvency.get("Debt-to-Equity")
    op_margin = profitability.get("Operating Margin")
    roe = profitability.get("ROE")

    is_bankrupt_risk = (
        not is_financial
        and debt_to_equity is not None
        and debt_to_equity > 250
    )
    is_losing_money = (
        not is_financial
        and op_margin is not None
        and op_margin < 0.02
    )
    is_fake_roe = roe is not None and roe > 1.0
    is_weak_financial_roe = is_financial and roe is not None and roe < 0.08

    return (
        isinstance(mos_normal, float)
        and mos_normal > 0.2
        and moat is not None
        and moat >= 3
        and not is_bankrupt_risk
        and not is_losing_money
        and not is_fake_roe
        and not is_weak_financial_roe
        and passes_peer_robustness(analysis.get("peers"))
    )


def save_analysis_txt(analysis):
    """
    Saves the full analysis as a human-readable .txt under
    analisis-accion/<TICKER>/<YYYY-MM-DD>.txt
    """
    try:
        ticker = analysis.get("ticker")
        base_dir = os.path.join(os.getcwd(), "analisis-accion", ticker)
        os.makedirs(base_dir, exist_ok=True)
        file_path = os.path.join(base_dir, datetime.now().strftime("%Y-%m-%d") + ".txt")
        cur = analysis.get("currency_symbol") or "$"
        model = analysis.get("valuation_model") or "DCF"

        lines = []
        lines.append("=" * 80)
        lines.append(f"ANALISIS DE {analysis.get('company_name', ticker)} - {ticker}")
        lines.append("=" * 80)
        if isinstance(analysis.get("price"), (int, float)):
            lines.append(f"Precio actual: {cur}{analysis['price']:.2f}")
        else:
            lines.append("Precio actual: N/A")
        lines.append(f"MOAT Score: {analysis.get('moat_score', 'N/A')}/7")
        lines.append(
            f"Industry: {analysis.get('industry', 'N/A')} | Sector: {analysis.get('sector', 'N/A')}"
        )
        lines.append(f"Modelo de valoracion: {model}")
        if analysis.get("is_financial"):
            lines.append(
                "Nota: Sector financiero/asegurador — FCF-DCF descartado; "
                "se usa DDM (Gordon) + Excess Returns + P/E normalizado."
            )
        lines.append("")

        lines.append("--- VALUATION ---")
        for key, value in (analysis.get("valuation") or {}).items():
            if isinstance(value, (int, float)):
                if "Price/FCF" in key or key in ("P/E Ratio", "P/BV Ratio", "EV/EBITDA"):
                    lines.append(f"{key}: {value:.2f}" if "Price/FCF" not in key else f"{key}: {value:.1f}x")
                elif key in ("EPS (normalized)", "BVPS"):
                    lines.append(f"{key}: {cur}{value:.2f}")
                else:
                    lines.append(f"{key}: {value:.2f}")
            else:
                lines.append(f"{key}: N/A")

        lines.append("")
        lines.append("--- SOLVENCY & HEALTH ---")
        for key, value in (analysis.get("solvency") or {}).items():
            if key == "Note":
                lines.append(f"{key}: {value}")
            elif key == "Net Debt" and isinstance(value, (int, float)):
                lines.append(f"{key}: {cur}{value:,.0f}")
            else:
                lines.append(
                    f"{key}: {value:.2f}" if isinstance(value, (int, float)) else f"{key}: N/A"
                )

        lines.append("")
        lines.append("--- PROFITABILITY & EFFICIENCY ---")
        for key, value in (analysis.get("profitability") or {}).items():
            if isinstance(value, (int, float)) and value == value:  # reject NaN
                lines.append(f"{key}: {value:.2%}")
            else:
                lines.append(f"{key}: N/A")

        lines.append("")
        lines.append("--- CASH FLOW / SHAREHOLDER YIELD ---")
        for key, value in (analysis.get("cash_flow") or {}).items():
            if isinstance(value, (int, float)):
                if "Yield" in key:
                    val = value / 100 if value > 1 else value
                    lines.append(f"{key}: {val:.2%}")
                elif "share" in key.lower() or "FCF" in key or "Free Cash" in key:
                    if "share" in key.lower():
                        lines.append(f"{key}: {cur}{value:.2f}")
                    else:
                        lines.append(f"{key}: {cur}{value:,.0f}")
                else:
                    lines.append(f"{key}: {value}")
            else:
                lines.append(f"{key}: N/A")

        details = analysis.get("financial_iv_details") or {}
        if details.get("Normal"):
            d = details["Normal"]
            lines.append("")
            lines.append("--- FINANCIAL MODEL BREAKDOWN (Normal) ---")
            if d.get("DDM") is not None:
                lines.append(f"DDM (Gordon): {cur}{d['DDM']:.2f}  [Ke={d['Ke']:.1%}, g={d['g']:.1%}]")
            if d.get("ExcessReturns") is not None:
                lines.append(f"Excess Returns (P/B): {cur}{d['ExcessReturns']:.2f}")
            if d.get("PE_norm") is not None:
                lines.append(f"P/E normalizado: {cur}{d['PE_norm']:.2f}")

        lines.append("")
        lines.append("--- PEERS / INDUSTRIA ---")
        lines.extend(format_peer_section(analysis.get("peers")))

        lines.append("")
        lines.append("--- INTRINSIC VALUE & MARGIN OF SAFETY ---")
        for scenario in SCENARIO_NAMES:
            iv = analysis.get("intrinsic_value", {}).get(scenario)
            mos = analysis.get("margin_of_safety", {}).get(scenario)
            if isinstance(iv, (int, float)) and isinstance(mos, (int, float)):
                lines.append(f"{scenario:<18}: IV {cur}{iv:>8.2f} | MOS {mos:>8.2%}")
            else:
                iv_text = f"{cur}{iv:.2f}" if isinstance(iv, (int, float)) else "N/A"
                mos_text = f"{mos:.2%}" if isinstance(mos, (int, float)) else "N/A"
                lines.append(f"{scenario:<18}: IV {iv_text} | MOS {mos_text}")

        reason = analysis.get("error_reason")
        if reason:
            lines.append("")
            lines.append(f"Aviso: {reason}")

        if is_quality_gem(analysis):
            lines.append("")
            lines.append(">>> Cumple criterios de GEMA (MOS + moat + peers).")
        elif (analysis.get("peers") or {}).get("is_relative_trap"):
            lines.append("")
            lines.append(">>> Descarta por trampa relativa vs industria.")

        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        print(f"{Colors.GREEN}>>> Informe guardado en: {file_path}{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}Failed to save analysis TXT: {e}{Colors.RESET}")


def print_single_ticker_report(analysis):
    """
    Prints a detailed report for a single ticker in terminal and saves it to disk.
    """
    if not analysis:
        print(f"{Colors.RED}No se pudo analizar el ticker solicitado.{Colors.RESET}")
        return

    cur = analysis.get("currency_symbol") or "$"
    model = analysis.get("valuation_model") or "DCF"

    print("\n" + "=" * 70)
    company_name = analysis.get('company_name', analysis['ticker'])
    print(f"ANALISIS INDIVIDUAL: {company_name} ({analysis['ticker']})")
    print("=" * 70)

    if isinstance(analysis.get("price"), (int, float)):
        print(f"Precio actual: {cur}{analysis['price']:.2f}")
    else:
        print("Precio actual: N/A")

    print(f"MOAT Score: {analysis.get('moat_score', 'N/A')}/7")
    if analysis.get("industry"):
        print(f"Industry: {analysis.get('industry')} | Sector: {analysis.get('sector', 'N/A')}")
    print(f"Modelo de valoracion: {model}")
    if analysis.get("is_financial"):
        print(
            f"{Colors.YELLOW}Sector financiero/asegurador: DCF-FCF descartado -> "
            f"DDM + Excess Returns{Colors.RESET}"
        )

    print("\n--- VALOR INTRINSECO Y MARGEN DE SEGURIDAD ---")
    for scenario in SCENARIO_NAMES:
        iv = analysis.get("intrinsic_value", {}).get(scenario)
        mos = analysis.get("margin_of_safety", {}).get(scenario)
        iv_text = f"{cur}{iv:.2f}" if isinstance(iv, (int, float)) else str(iv)
        mos_text = f"{mos:.2%}" if isinstance(mos, (int, float)) else "N/A"
        print(f"{scenario:<18}: IV {iv_text:>12} | MOS {mos_text:>8}")

    details = (analysis.get("financial_iv_details") or {}).get("Normal")
    if details:
        print("\n--- DESGLOSE MODELO FINANCIERO (Normal) ---")
        if details.get("DDM") is not None:
            print(f"DDM (Gordon):          {cur}{details['DDM']:.2f}  (Ke={details['Ke']:.1%}, g={details['g']:.1%})")
        if details.get("ExcessReturns") is not None:
            print(f"Excess Returns (P/B):  {cur}{details['ExcessReturns']:.2f}")
        if details.get("PE_norm") is not None:
            print(f"P/E normalizado:       {cur}{details['PE_norm']:.2f}")

    print("\n--- PEERS / INDUSTRIA ---")
    for line in format_peer_section(analysis.get("peers")):
        print(line)

    valuation = analysis.get("valuation") or {}
    if isinstance(valuation.get("Price/FCF"), (int, float)):
        print(f"Price/FCF (company): {valuation['Price/FCF']:.1f}x")
    if isinstance(valuation.get("P/E Ratio"), (int, float)) and analysis.get("is_financial"):
        print(f"P/E (company): {valuation['P/E Ratio']:.1f}x")
    if isinstance(valuation.get("P/BV Ratio"), (int, float)) and analysis.get("is_financial"):
        print(f"P/B (company): {valuation['P/BV Ratio']:.2f}x")

    reason = analysis.get("error_reason")
    if reason:
        print(f"\n{Colors.YELLOW}Aviso: {reason}{Colors.RESET}")

    if is_quality_gem(analysis):
        print(f"\n{Colors.GREEN}>>> Cumple criterios de GEMA (MOS + moat + peers).{Colors.RESET}")
    elif analysis.get("peers", {}).get("is_relative_trap"):
        print(f"\n{Colors.YELLOW}>>> Descarta por trampa relativa vs industria.{Colors.RESET}")

    save_analysis_txt(analysis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analizador Value Investing: mercado completo o ticker individual."
    )
    parser.add_argument(
        "-t",
        "--ticker",
        type=str,
        help="Ticker individual a analizar (ejemplo: AAPL o SAN.MC).",
    )
    args = parser.parse_args()

    print("\n" + "="*50)
    print("--- Starting Value Investing Analysis ---")
    print("="*50)
    
    start_time = time.time()

    if args.ticker:
        ticker = args.ticker.strip().upper()
        print(f"--> Analizando ticker individual: {ticker}")
        single_result = analyze_ticker(ticker)
        if single_result:
            benchmarks = load_benchmarks()
            if benchmarks:
                single_result["peers"] = compare_to_peers(single_result, benchmarks)
                print("--> Benchmarks de industria cargados desde data/industry_benchmarks.json")
            else:
                single_result["peers"] = compare_to_peers(single_result, None)
                print(
                    f"{Colors.YELLOW}--> Sin benchmarks locales. Ejecuta un scan completo "
                    f"para habilitar peers vs industria.{Colors.RESET}"
                )
        print_single_ticker_report(single_result)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\n" + "=" * 50)
        print(f"Tiempo total: {int(execution_time // 60)} min {int(execution_time % 60)} seg.")
        print("=" * 50)
        raise SystemExit(0)

    # --- CARGA DE TICKERS ---
    market_files = [os.path.join("data", f) for f in os.listdir("data") if f.endswith(".csv")]
    all_tickers = []
    for f in market_files:
        tickers, _ = get_tickers_from_csv(f)
        all_tickers.extend(tickers)
    
    unique_tickers = sorted(list(set(all_tickers)))
    print(f"--> Analyzing {len(unique_tickers)} unique tickers.")
    
    # --- INICIO DEL ANÁLISIS EN PARALELO (pasada 1: datos + DCF/moat) ---
    all_analyses = []
    total_tickers = len(unique_tickers)
    start_time = time.time()

    print(f"--> Analizando {total_tickers} tickers usando 5 hilos simultáneos...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {executor.submit(analyze_ticker, t): t for t in unique_tickers}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_ticker), 1):
            ticker = future_to_ticker[future]
            try:
                analysis = future.result()
                if analysis:
                    all_analyses.append(analysis)
                    mos_normal = analysis['margin_of_safety'].get("Normal")
                    moat = analysis['moat_score']
                    reason = analysis.get("error_reason")
                    industry = analysis.get("industry") or "?"
                    
                    if isinstance(mos_normal, (float, int)):
                        mos_str = f"{mos_normal:.2%}"
                    else:
                        mos_str = f"{Colors.YELLOW}{reason if reason else 'N/A'}{Colors.RESET}"
                    
                    print(
                        f"[{i}/{total_tickers}] Analyzed: {ticker:<8} | Moat: {moat} | "
                        f"MOS: {mos_str} | {industry}"
                    )
            except Exception as e:
                print(f"{Colors.RED}Error con {ticker}: {e}{Colors.RESET}")

    # --- PASADA 2: benchmarks de industria + filtro de gemas con peers ---
    print(f"\n--> Construyendo benchmarks de industria a partir de {len(all_analyses)} análisis...")
    benchmarks = build_industry_benchmarks(all_analyses)
    save_benchmarks(benchmarks)
    usable_industries = sum(
        1
        for ind in benchmarks.get("industries", {}).values()
        if (ind.get("sample_size") or 0) >= benchmarks.get("peer_min_sample", 5)
    )
    print(
        f"--> {len(benchmarks.get('industries', {}))} industrias "
        f"({usable_industries} con muestra suficiente). Guardado en data/industry_benchmarks.json"
    )

    undervalued_opportunities = []
    relative_traps = 0
    for analysis in all_analyses:
        analysis["peers"] = compare_to_peers(analysis, benchmarks)
        if analysis["peers"].get("is_relative_trap"):
            relative_traps += 1
        if is_quality_gem(analysis):
            undervalued_opportunities.append(analysis)

    print(
        f"--> Peer filter: {relative_traps} trampas relativas detectadas; "
        f"{len(undervalued_opportunities)} gemas tras peers."
    )

    # --- GENERACIÓN DEL ARCHIVO FINAL ---
    if undervalued_opportunities:
        # Mejor margen de seguridad primero; a igualdad, más ventaja vs peers
        undervalued_opportunities.sort(
            key=lambda x: (
                x["margin_of_safety"]["Normal"],
                (x.get("peers") or {}).get("edge_score") or 0,
            ),
            reverse=True,
        )
        
        output_dir = ".\\infravaloradas"
        os.makedirs(output_dir, exist_ok=True)
        today_str = datetime.now().strftime("%Y-%m-%d")
        output_file = os.path.join(output_dir, f"infravaloradas_{today_str}.txt")
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=================================================================================\n")
            f.write(f"RESUMEN DE ACCIONES INFRAVALORADAS - {today_str}\n")
            f.write("Filtro: MOS>20% + Moat>=3 + red flags + peers (sin trampa relativa vs industria)\n")
            f.write("Financieras/aseguradoras: modelo DDM+ExcessReturns (no FCF-DCF)\n")
            f.write("=================================================================================\n")
            f.write(
                f"{'Ticker':<10} | {'Precio':<10} | {'V.I. Normal':<12} | {'Margen (MOS)':<12} | "
                f"{'MOAT':<6} | {'Peer':<8} | Industry\n"
            )
            f.write("-" * 110 + "\n")
            for op in undervalued_opportunities:
                peers = op.get("peers") or {}
                edge = f"{peers.get('edge_score', 0)}/{peers.get('max_edges', 0)}"
                industry = (peers.get("industry") or op.get("industry") or "")[:28]
                cur = op.get("currency_symbol") or "$"
                f.write(
                    f"{op['ticker']:<10} | {cur}{op['price']:<9.2f} | "
                    f"{cur}{op['intrinsic_value']['Normal']:<11.2f} | "
                    f"{op['margin_of_safety']['Normal']:<12.2%} | "
                    f"{op['moat_score']}/7   | {edge:<8} | {industry}\n"
                )

            f.write("\n\n")

            f.write("=================================================================================\n")
            f.write("DETALLE EXTENDIDO DE CADA OPORTUNIDAD\n")
            f.write("=================================================================================\n")

            for analysis in undervalued_opportunities:
                cur = analysis.get("currency_symbol") or "$"
                model = analysis.get("valuation_model") or "DCF"
                f.write(f"\n\n{'#'*60}\n")
                f.write(f"### ANÁLISIS DE {analysis['ticker']}\n")
                f.write(f"{'#'*60}\n")
                f.write(f"Current Price: {cur}{analysis['price']:.2f}\n")
                f.write(f"MOAT Score: {analysis['moat_score']}/7\n")
                f.write(
                    f"Industry: {analysis.get('industry', 'N/A')} | "
                    f"Sector: {analysis.get('sector', 'N/A')}\n"
                )
                f.write(f"Modelo de valoracion: {model}\n")

                f.write("\n--- VALUATION ---\n")
                for key, value in analysis["valuation"].items():
                    if isinstance(value, (int, float)):
                        if "Price/FCF" in key:
                            f.write(f"{key}: {value:.1f}x\n")
                        elif key in ("EPS (normalized)", "BVPS"):
                            f.write(f"{key}: {cur}{value:.2f}\n")
                        else:
                            f.write(f"{key}: {value:.2f}\n")
                    else:
                        f.write(f"{key}: N/A\n")

                f.write("\n--- SOLVENCY & HEALTH ---\n")
                for key, value in analysis["solvency"].items():
                    if key == "Note":
                        f.write(f"{key}: {value}\n")
                    elif key == "Net Debt" and isinstance(value, (int, float)):
                        f.write(f"{key}: {cur}{value:,.0f}\n")
                    else:
                        f.write(
                            f"{key}: {value:.2f}\n"
                            if isinstance(value, (int, float))
                            else f"{key}: N/A\n"
                        )

                f.write("\n--- PROFITABILITY & EFFICIENCY ---\n")
                for key, value in analysis["profitability"].items():
                    if isinstance(value, (int, float)) and value == value:
                        f.write(f"{key}: {value:.2%}\n")
                    else:
                        f.write(f"{key}: N/A\n")

                f.write("\n--- CASH FLOW / SHAREHOLDER YIELD ---\n")
                for key, value in analysis["cash_flow"].items():
                    if isinstance(value, (int, float)):
                        if "Yield" in key:
                            val = value / 100 if value > 1 else value
                            f.write(f"{key}: {val:.2%}\n")
                        elif "share" in key.lower():
                            f.write(f"{key}: {cur}{value:.2f}\n")
                        elif "FCF" in key or "Free Cash" in key:
                            f.write(f"{key}: {cur}{value:,.0f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                    else:
                        f.write(f"{key}: N/A\n")

                details = (analysis.get("financial_iv_details") or {}).get("Normal")
                if details:
                    f.write("\n--- FINANCIAL MODEL BREAKDOWN (Normal) ---\n")
                    if details.get("DDM") is not None:
                        f.write(
                            f"DDM (Gordon): {cur}{details['DDM']:.2f}  "
                            f"[Ke={details['Ke']:.1%}, g={details['g']:.1%}]\n"
                        )
                    if details.get("ExcessReturns") is not None:
                        f.write(f"Excess Returns (P/B): {cur}{details['ExcessReturns']:.2f}\n")
                    if details.get("PE_norm") is not None:
                        f.write(f"P/E normalizado: {cur}{details['PE_norm']:.2f}\n")

                f.write("\n--- PEERS / INDUSTRIA ---\n")
                for line in format_peer_section(analysis.get("peers")):
                    f.write(line + "\n")

                f.write("\n--- INTRINSIC VALUE & MARGIN OF SAFETY ---\n")
                for scenario in SCENARIO_NAMES:
                    iv = analysis["intrinsic_value"].get(scenario)
                    mos = analysis["margin_of_safety"].get(scenario)
                    if isinstance(iv, (int, float)) and isinstance(mos, (int, float)):
                        f.write(f"{scenario:<18}: IV {cur}{iv:>8.2f} | MOS {mos:>8.2%}\n")
                    else:
                        iv_text = f"{cur}{iv:.2f}" if isinstance(iv, (int, float)) else "N/A"
                        mos_text = f"{mos:.2%}" if isinstance(mos, (int, float)) else "N/A"
                        f.write(f"{scenario:<18}: IV {iv_text} | MOS {mos_text}\n")

        print(f"\n{Colors.GREEN}>>> Análisis finalizado. {len(undervalued_opportunities)} oportunidades encontradas.")
        print(f">>> Informe detallado generado en: {output_file}{Colors.RESET}")
    else:
        print(f"\n{Colors.RED}>>> No se encontraron acciones que cumplan los criterios.{Colors.RESET}")

    # --- TIEMPO DE EJECUCIÓN ---
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n" + "="*50)
    print(f"Tiempo total: {int(execution_time // 60)} min {int(execution_time % 60)} seg.")
    print("="*50)