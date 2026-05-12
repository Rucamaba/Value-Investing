"""
This script analyzes tickers to identify buying opportunities based on the Value Investing strategy.
"""

import yfinance as yf
import pandas as pd
import os
import numpy as np
import argparse
from markets import get_tickers_from_csv
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

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'

import random
import time

MAX_PERPETUAL_GROWTH_RATE = 0.03


def normalize_sector(sector):
    """Map the raw Yahoo sector into a small set of investing buckets."""
    if not isinstance(sector, str):
        return "unknown"

    sector_lower = sector.strip().lower()
    if not sector_lower:
        return "unknown"
    if "technology" in sector_lower:
        return "technology"
    if "health" in sector_lower:
        return "healthcare"
    if "financial" in sector_lower:
        return "financials"
    if "real estate" in sector_lower or "reit" in sector_lower:
        return "reit"
    if "utility" in sector_lower:
        return "utilities"
    if "consumer staple" in sector_lower or "staple" in sector_lower:
        return "staples"
    if "consumer cyclical" in sector_lower:
        return "consumer_cyclical"
    if (
        "energy" in sector_lower
        or "materials" in sector_lower
        or "basic materials" in sector_lower
        or "commodity" in sector_lower
    ):
        return "commodities"
    if "industrial" in sector_lower:
        return "industrials"

    return "other"


def calculate_intrinsic_value_by_multiples(book_value_per_share, target_pb_by_scenario, roe=None):
    """Fallback valuation for sectors where FCF-based DCF is not reliable."""
    if not isinstance(book_value_per_share, (int, float)) or book_value_per_share <= 0:
        return {scenario: "N/A" for scenario in target_pb_by_scenario}

    if isinstance(roe, (int, float)) and roe < 0:
        target_pb_by_scenario = {
            scenario: max(target_pb - 0.2, 0.1)
            for scenario, target_pb in target_pb_by_scenario.items()
        }

    return {
        scenario: book_value_per_share * target_pb
        for scenario, target_pb in target_pb_by_scenario.items()
    }

def get_financial_data(ticker_symbol):
    """
    Downloads and returns financial data for a given ticker with a retry mechanism.
    Updated to let yfinance handle its own internal session (curl_cffi).
    """
    for attempt in range(3): 
        try:
            # 1. Pausa aleatoria ANTES de pedir datos (crucial para 8 hilos)
            # Aumentamos un poco el rango para dar respiro a la API
            time.sleep(random.uniform(2.0, 3.5))
            
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

def score_moat(info, financials, cashflow, balance_sheet, total_debt, cash, sector_group=None, roe=None, gross_margin=None):
    """
    Scores the company's competitive moat based on profitability and consistency.
    """
    score = 0
    
    # 1. High Profitability: sector-aware ROIC / ROE
    if sector_group == "financials":
        if isinstance(roe, (int, float)) and roe > 0.12:
            score += 2
    elif sector_group == "reit":
        if isinstance(roe, (int, float)) and roe > 0.08:
            score += 2
    else:
        try:
            # NOPAT (EBIT * (1 - Tax Rate)) simplificado
            ebit = financials.loc['EBIT'].iloc[0]
            # Capital Invertido = Deuda + Patrimonio - Efectivo
            cap_inv = total_debt + balance_sheet.loc['Stockholders Equity'].iloc[0] - cash
            if cap_inv > 0:
                roic_calc = (ebit * 0.75) / cap_inv # Asumimos 25% de impuestos
                if roic_calc > 0.15: score += 2
        except:
            pass

    if sector_group == "technology" and isinstance(gross_margin, (int, float)) and gross_margin > 0.50:
        score += 1
        
    # 2. Margin Stability (Stable/Growing Operating Margin over 3 years)
    try:
        op_income = financials.loc['Operating Income']
        revenue = financials.loc['Total Revenue']
        margins = op_income / revenue
        if len(margins) >= 3 and margins.iloc[0] >= margins.iloc[-1]:
            score += 2
    except KeyError:
        pass

    # 3. Low Reinvestment Needs (CapEx / CFO < 30%)
    try:
        cfo = cashflow.loc['Total Cash From Operating Activities'].iloc[0]
        capex = cashflow.loc['Capital Expenditures'].iloc[0]
        if cfo > 0 and (abs(capex) / cfo) < 0.30:
            score += 1
    except (KeyError, IndexError):
        pass

    # 4. Consistent EPS Growth
    try:
        eps = financials.loc['Basic EPS']
        if len(eps) >= 3 and eps.iloc[0] > eps.iloc[-1]:
            score += 1
    except KeyError:
        pass

    # 5. Capacidad de pago de intereses (Interest Coverage > 5)
    try:
        ebit = financials.loc['EBIT'].iloc[0]
        interest_exp = abs(financials.loc['Interest Expense'].iloc[0])
        if interest_exp > 0 and (ebit / interest_exp) > 5:
            score += 1
    except:
        pass

    return score

def calculate_intrinsic_value(fcf_current, g_ultra_pessimistic, g_pessimistic, g_normal, g_optimistic, g_ultra_optimistic,
                              shares_outstanding, cash, total_debt, current_price, sector_group=None, discount_rate_normal=None):
    """
    Calculates the intrinsic value using a DCF model for pessimistic, normal, and optimistic scenarios.
    """
    perpetual_growth_rate = min(PERPETUAL_GROWTH_RATE, MAX_PERPETUAL_GROWTH_RATE)

    if discount_rate_normal is None:
        if sector_group in {"utilities", "staples"}:
            discount_rate_normal = 0.08
        elif sector_group in {"technology", "commodities"}:
            discount_rate_normal = 0.12
        else:
            discount_rate_normal = DISCOUNT_RATE_NORMAL

    scenarios = {
        "Ultra Pessimistic": {"g": g_ultra_pessimistic, "r": DISCOUNT_RATE_ULTRA_PESSIMISTIC},
        "Pessimistic": {"g": g_pessimistic, "r": DISCOUNT_RATE_PESSIMISTIC},
        "Normal": {"g": g_normal, "r": discount_rate_normal},
        "Optimistic": {"g": g_optimistic, "r": DISCOUNT_RATE_OPTIMISTIC},
        "Ultra Optimistic": {"g": g_ultra_optimistic, "r": DISCOUNT_RATE_ULTRA_OPTIMISTIC},
    }
    
    results = {}

    for scenario_name, params in scenarios.items():
        g = params["g"]
        r = params["r"]

        if r <= perpetual_growth_rate:
            results[scenario_name] = "Invalid Discount Rate"
            continue

        if (r - perpetual_growth_rate) < 0.01:
            r = perpetual_growth_rate + 0.015

        fcf_projections = []
        last_fcf = fcf_current
        
        for _ in range(10):
            last_fcf *= (1 + g)
            fcf_projections.append(last_fcf)
            
        dcf = [fcf / ((1 + r) ** (i + 1)) for i, fcf in enumerate(fcf_projections)]
        
        fcf_year_10 = fcf_projections[-1]
        terminal_value = fcf_year_10 * (1 + perpetual_growth_rate) / (r - perpetual_growth_rate)
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
    sector_group = normalize_sector(sector)

    # --- Ratios de Valoración ---
    pe_ratio = info.get("trailingPE")
    pb_ratio = info.get("priceToBook")
    ev_to_ebitda = info.get("enterpriseToEbitda")

    # --- Solvencia y Salud Financiera ---
    debt_to_equity = info.get("debtToEquity")
    current_ratio = info.get("currentRatio")
    
    # MEJORA: Búsqueda flexible de Caja (Cash)
    cash_options = ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments', 'Cash']
    cash = 0
    for opt in cash_options:
        if opt in balance_sheet.index:
            cash = balance_sheet.loc[opt].iloc[0]
            break
            
    total_debt = info.get("totalDebt") or 0

    # Mejora para Solvencia en tickers europeos
    if current_ratio is None:
        try:
            # Intentamos buscar las etiquetas manuales en el Balance Sheet
            current_assets = balance_sheet.loc['Total Current Assets'].iloc[0]
            current_liabilities = balance_sheet.loc['Total Current Liabilities'].iloc[0]
            current_ratio = current_assets / current_liabilities
        except:
            pass

    # --- Rentabilidad y Eficiencia ---
    try:
        # ROE: Beneficio Neto / Capital de los Accionistas
        net_income = financials.loc['Net Income'].iloc[0]
        equity = balance_sheet.loc['Stockholders Equity'].iloc[0]
        roe = net_income / equity if equity > 0 else None
    except:
        roe = info.get("returnOnEquity") # Fallback al dato de Yahoo

    try:
        # ROIC: (EBIT * 0.75) / (Deuda + Equity - Caja)
        ebit = financials.loc['EBIT'].iloc[0]
        equity = balance_sheet.loc['Stockholders Equity'].iloc[0]
        # Usamos las variables 'total_debt' y 'cash' que ya calculaste arriba
        invested_capital = total_debt + equity - cash
        if invested_capital is None or invested_capital <= 0:
            roic = None
        else:
            roic = (ebit * 0.75) / invested_capital
    except:
        roic = info.get("returnOnInvestedCapital") # Fallback al dato de Yahoo

    try:
        ebit = financials.loc['EBIT'].iloc[0]
        interest_expense = abs(financials.loc['Interest Expense'].iloc[0])
        interest_coverage = ebit / interest_expense if interest_expense > 0 else None
    except:
        interest_coverage = info.get("interestCoverage")

    # EBITDA: EBIT + Depreciation & Amortization
    try:
        ebit = financials.loc['EBIT'].iloc[0]
        da = financials.loc['Depreciation And Amortization'].iloc[0]
        ebitda = ebit + da
    except:
        ebitda = info.get("ebitda") # Fallback al dato de Yahoo
    # EBITDA Growth (3Y): CAGR de EBITDA para los últimos 3 años
    try:
        ebitda_series = (financials.loc['EBIT'] + financials.loc['Depreciation And Amortization']).dropna()
        if len(ebitda_series) >= 4:
            ebitda_now = ebitda_series.iloc[0]
            ebitda_3y_ago = ebitda_series.iloc[3]

            # Solo calculamos si ambos son positivos para evitar errores matemáticos
            if ebitda_now > 0 and ebitda_3y_ago > 0:
                ebitda_growth_3y = (ebitda_now / ebitda_3y_ago) ** (1/3) - 1
            else:
                ebitda_growth_3y = None
        else:
            ebitda_growth_3y = None
    except:
        ebitda_growth_3y = None


    gross_margin = info.get("grossMargins")
    operating_margin = info.get("operatingMargins")

    # --- Flujo de Caja (Cálculo Robusto) ---
    try:
        if 'Free Cash Flow' in cashflow.index:
            fcf_series = cashflow.loc['Free Cash Flow'].dropna()
        else:
            cfo_key = next((x for x in ['Operating Cash Flow', 'Total Cash From Operating Activities'] if x in cashflow.index), None)
            capex_key = next((x for x in ['Capital Expenditure', 'Investing Cash Flow'] if x in cashflow.index), None)
            
            if cfo_key and capex_key:
                # Capex suele ser negativo en yfinance
                fcf_series = (cashflow.loc[cfo_key] - abs(cashflow.loc[capex_key])).dropna()
            else:
                fcf_series = pd.Series()

        fcf = fcf_series.iloc[:3].mean() if not fcf_series.empty else None
    except Exception:
        fcf = None

    dividend_yield = info.get("trailingAnnualDividendYield") or info.get("dividendYield")
    
    # --- Cálculo de Crecimiento (CAGR) ---
    try:
        revenues = financials.loc['Total Revenue'].dropna()
        if len(revenues) > 1:
            num_years = len(revenues) - 1
            cagr = (revenues.iloc[0] / revenues.iloc[-1]) ** (1/num_years) - 1
            if cagr < 0: cagr = 0.01 
            g_normal = min(cagr * 0.7, 0.15) 
        else:
            g_normal = 0.05

        # Definimos los 5 niveles de crecimiento
        g_ultra_pessimistic = g_normal * 0.3
        g_pessimistic = g_normal * 0.6
        g_optimistic = g_normal * 1.3
        g_ultra_optimistic = g_normal * 1.6
        
    except (KeyError, ZeroDivisionError, Exception):
        g_normal = 0.05
        g_ultra_pessimistic, g_pessimistic = 0.01, 0.03
        g_optimistic, g_ultra_optimistic = 0.07, 0.10
    
    shares_outstanding = info.get("impliedSharesOutstanding") or info.get("sharesOutstanding")
    current_price = info.get("currentPrice")
    book_value_per_share = info.get("bookValue")

    if not isinstance(book_value_per_share, (int, float)) or book_value_per_share <= 0:
        try:
            equity = balance_sheet.loc['Stockholders Equity'].iloc[0]
            if shares_outstanding and shares_outstanding > 0:
                book_value_per_share = equity / shares_outstanding
        except:
            if isinstance(current_price, (int, float)) and isinstance(pb_ratio, (int, float)) and pb_ratio > 0:
                book_value_per_share = current_price / pb_ratio

    # --- Cálculo del Valor Intrínseco ---
    error_reason = None
    if sector_group in {"financials", "reit"}:
        valuation_model = "Multiples"
        if sector_group == "financials":
            target_pb_by_scenario = {
                "Ultra Pessimistic": 0.90,
                "Pessimistic": 1.00,
                "Normal": 1.20,
                "Optimistic": 1.35,
                "Ultra Optimistic": 1.50,
            }
        else:
            target_pb_by_scenario = {
                "Ultra Pessimistic": 0.80,
                "Pessimistic": 0.90,
                "Normal": 1.05,
                "Optimistic": 1.15,
                "Ultra Optimistic": 1.25,
            }

        if isinstance(dividend_yield, (int, float)) and dividend_yield >= 0.04:
            target_pb_by_scenario = {scenario: target_pb + 0.05 for scenario, target_pb in target_pb_by_scenario.items()}

        if isinstance(roe, (int, float)) and roe < 0:
            target_pb_by_scenario = {scenario: target_pb - 0.2 for scenario, target_pb in target_pb_by_scenario.items()}
        elif isinstance(roe, (int, float)) and roe >= 0.10:
            target_pb_by_scenario = {scenario: target_pb + 0.05 for scenario, target_pb in target_pb_by_scenario.items()}

        intrinsic_values = calculate_intrinsic_value_by_multiples(book_value_per_share, target_pb_by_scenario, roe=roe)
        if all(value == "N/A" for value in intrinsic_values.values()):
            error_reason = "No Book Value Data"
    else:
        valuation_model = "DCF"
        if fcf is None or fcf <= 0:
            error_reason = "Negative/Zero FCF"
        elif not shares_outstanding:
            error_reason = "No Shares Data"

        if error_reason:
            intrinsic_values = {sc: "N/A" for sc in ["Ultra Pessimistic", "Pessimistic", "Normal", "Optimistic", "Ultra Optimistic"]}
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
                sector_group=sector_group
            )
            # Si la función devuelve un string (como "Check Data"), lo capturamos
            if isinstance(intrinsic_values.get("Normal"), str):
                error_reason = intrinsic_values.get("Normal")
    
    margin_of_safety = {}
    if current_price:
        for scenario, iv in intrinsic_values.items():
            if isinstance(iv, (int, float)) and iv > 0:
                margin_of_safety[scenario] = (iv - current_price) / iv
            else:
                margin_of_safety[scenario] = None
    
    # Moat score usando los nuevos datos de cash y total_debt
    moat_score = score_moat(info, financials, cashflow, balance_sheet, total_debt, cash, sector_group=sector_group, roe=roe, gross_margin=gross_margin)

    return {
        "ticker": ticker_symbol,
        "price": current_price,
        "error_reason": error_reason,
        "sector": sector,
        "sector_group": sector_group,
        "valuation_model": valuation_model,
        "moat_score": moat_score,
        "valuation": {
            "P/E Ratio": pe_ratio,
            "P/BV Ratio": pb_ratio,
            "EV/EBITDA": ev_to_ebitda,
        },
        "solvency": {
            "Debt-to-Equity": debt_to_equity,
            "Current Ratio": current_ratio,
            "Interest Coverage": interest_coverage,
            "Net Debt": total_debt - cash,
        },
        "profitability": {
            "ROE": roe,
            "ROIC": roic,
            "EBITDA": ebitda,
            "EBITDA Growth (3Y)": ebitda_growth_3y,
            "Gross Margin": gross_margin,
            "Operating Margin": operating_margin,
        },
        "cash_flow": {
            "Free Cash Flow (3Y Avg)": fcf,
            "Dividend Yield": dividend_yield,
        },
        "market_data": {
            "Fifty Two Week High": info.get("fiftyTwoWeekHigh"),
        },
        "intrinsic_value": intrinsic_values,
        "margin_of_safety": margin_of_safety,
        "company_name": info.get("longName"),
    }


def save_analysis_txt(analysis):
    """
    Saves the full analysis as a human-readable .txt under analisis-accion/<TICKER>/<YYYY-MM-DD>.txt
    """
    try:
        ticker = analysis.get('ticker')
        base_dir = os.path.join(os.getcwd(), "analisis-accion", ticker)
        os.makedirs(base_dir, exist_ok=True)
        file_path = os.path.join(base_dir, datetime.now().strftime("%Y-%m-%d") + ".txt")

        lines = []
        reason = analysis.get('error_reason')
        if reason:
            lines.append("!" * 80)
            lines.append("!" + " " * 78 + "!")
            lines.append(f"!  ADVERTENCIA: {reason[:68]:<68}  !")
            lines.append("!" + " " * 78 + "!")
            lines.append("!  La valoración para este ticker puede ser poco fiable.            !")
            lines.append("!" * 80)
            lines.append("")
        lines.append("=" * 80)
        lines.append(f"ANALISIS DE {analysis.get('company_name', ticker)} - {ticker}")
        lines.append("=" * 80)
        lines.append(f"Precio actual: ${analysis.get('price'):.2f}" if isinstance(analysis.get('price'), (int, float)) else "Precio actual: N/A")
        lines.append(f"MOAT Score: {analysis.get('moat_score', 'N/A')}/7")
        lines.append(f"Sector: {analysis.get('sector') or analysis.get('sector_group') or 'N/A'}")
        lines.append("")

        lines.append("--- VALUATION ---")
        for key, value in analysis.get('valuation', {}).items():
            lines.append(f"{key}: {value:.2f}" if isinstance(value, (int, float)) else f"{key}: N/A")

        lines.append("")
        lines.append("--- SOLVENCY & HEALTH ---")
        for key, value in analysis.get('solvency', {}).items():
            if key == 'Net Debt' and isinstance(value, (int, float)):
                lines.append(f"{key}: ${value:,.0f}")
            else:
                lines.append(f"{key}: {value:.2f}" if isinstance(value, (int, float)) else f"{key}: N/A")

        lines.append("")
        lines.append("--- PROFITABILITY & EFFICIENCY ---")
        for key, value in analysis.get('profitability', {}).items():
            if key == 'EBITDA' and isinstance(value, (int, float)):
                lines.append(f"{key}: ${value:,.0f}")
            elif key == 'EBITDA Growth (3Y)' and isinstance(value, (int, float)):
                lines.append(f"{key}: {value:.2%}")
            else:
                lines.append(f"{key}: {value:.2%}" if isinstance(value, (int, float)) else f"{key}: N/A")

        lines.append("")
        lines.append("--- CASH FLOW ---")
        for key, value in analysis.get('cash_flow', {}).items():
            if 'Free Cash Flow' in key and isinstance(value, (int, float)):
                lines.append(f"{key}: ${value:,.0f}")
            elif 'Yield' in key and isinstance(value, (int, float)):
                val = value/100 if value > 1 else value
                lines.append(f"{key}: {val:.2%}")
            else:
                lines.append(f"{key}: N/A")

        lines.append("")
        lines.append("--- INTRINSIC VALUE & MARGIN OF SAFETY ---")
        for scenario in ["Ultra Pessimistic", "Pessimistic", "Normal", "Optimistic", "Ultra Optimistic"]:
            iv = analysis.get('intrinsic_value', {}).get(scenario)
            mos = analysis.get('margin_of_safety', {}).get(scenario)
            if isinstance(iv, (int, float)) and isinstance(mos, (int, float)):
                lines.append(f"{scenario:<18}: IV ${iv:>8.2f} | MOS {mos:>8.2%}")
            else:
                iv_text = f"${iv:.2f}" if isinstance(iv, (int, float)) else "N/A"
                mos_text = f"{mos:.2%}" if isinstance(mos, (int, float)) else "N/A"
                lines.append(f"{scenario:<18}: IV {iv_text} | MOS {mos_text}")

        if reason:
            lines.append("")
            lines.append(f"Aviso: {reason}")

        with open(file_path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(lines))

        print(f"{Colors.GREEN}Saved analysis TXT: {file_path}{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}Failed to save analysis TXT: {e}{Colors.RESET}")


def print_single_ticker_report(analysis):
    """
    Prints a detailed report for a single ticker in terminal.
    """
    if not analysis:
        print(f"{Colors.RED}No se pudo analizar el ticker solicitado.{Colors.RESET}")
        return

    print("\n" + "=" * 70)
    company_name = analysis.get('company_name', analysis['ticker'])
    print(f"ANALISIS INDIVIDUAL: {company_name} ({analysis['ticker']})")
    print("=" * 70)

    if isinstance(analysis.get("price"), (int, float)):
        print(f"Precio actual: ${analysis['price']:.2f}")
    else:
        print("Precio actual: N/A")

    print(f"MOAT Score: {analysis.get('moat_score', 'N/A')}/7")

    print("\n--- VALUATION ---")
    for key, value in analysis.get('valuation', {}).items():
        print(f"{key}: {value:.2f}" if isinstance(value, (int, float)) else f"{key}: N/A")

    print("\n--- SOLVENCY & HEALTH ---")
    for key, value in analysis.get('solvency', {}).items():
        if key == 'Net Debt' and isinstance(value, (int, float)):
            print(f"{key}: ${value:,.0f}")
        else:
            print(f"{key}: {value:.2f}" if isinstance(value, (int, float)) else f"{key}: N/A")

    print("\n--- PROFITABILITY & EFFICIENCY ---")
    for key, value in analysis.get('profitability', {}).items():
        if key == 'EBITDA' and isinstance(value, (int, float)):
            print(f"{key}: ${value:,.0f}")
        elif key == 'EBITDA Growth (3Y)' and isinstance(value, (int, float)):
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value:.2%}" if isinstance(value, (int, float)) else f"{key}: N/A")

    print("\n--- CASH FLOW ---")
    for key, value in analysis.get('cash_flow', {}).items():
        if 'Free Cash Flow' in key and isinstance(value, (int, float)):
            print(f"{key}: ${value:,.0f}")
        elif 'Yield' in key and isinstance(value, (int, float)):
            val = value/100 if value > 1 else value
            print(f"{key}: {val:.2%}")
        else:
            print(f"{key}: N/A")

    print("\n--- INTRINSIC VALUE & MARGIN OF SAFETY ---")
    for scenario in ["Ultra Pessimistic", "Pessimistic", "Normal", "Optimistic", "Ultra Optimistic"]:
        iv = analysis.get('intrinsic_value', {}).get(scenario)
        mos = analysis.get('margin_of_safety', {}).get(scenario)
        if isinstance(iv, (int, float)) and isinstance(mos, (int, float)):
            print(f"{scenario:<18}: IV ${iv:>8.2f} | MOS {mos:>8.2%}")
        else:
            iv_text = f"${iv:.2f}" if isinstance(iv, (int, float)) else "N/A"
            mos_text = f"{mos:.2%}" if isinstance(mos, (int, float)) else "N/A"
            print(f"{scenario:<18}: IV {iv_text} | MOS {mos_text}")

    reason = analysis.get('error_reason')
    if reason:
        print(f"\n{Colors.YELLOW}Aviso: {reason}{Colors.RESET}")

    # Save the full analysis as TXT in analisis-accion/<TICKER>/<date>.txt
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
    
    # Lista para almacenar las "joyas" encontradas
    undervalued_opportunities = []

    # --- INICIO DEL ANÁLISIS EN PARALELO ---
    undervalued_opportunities = []
    total_tickers = len(unique_tickers)
    start_time = time.time()

    print(f"--> Analizando {total_tickers} tickers usando 8 hilos simultáneos...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Lanzamos todas las tareas
        future_to_ticker = {executor.submit(analyze_ticker, t): t for t in unique_tickers}
        
        # Procesamos los resultados conforme van terminando
        for i, future in enumerate(concurrent.futures.as_completed(future_to_ticker), 1):
            ticker = future_to_ticker[future]
            try:
                analysis = future.result()
                if analysis:
                    # --- MONITORIZACIÓN POR TERMINAL CORREGIDA ---
                    mos_normal = analysis['margin_of_safety'].get("Normal")
                    moat = analysis['moat_score']
                    reason = analysis.get("error_reason")
                    
                    # Preparamos el texto del margen para evitar el error de formato
                    if isinstance(mos_normal, (float, int)):
                        mos_str = f"{mos_normal:.2%}"
                    else:
                        mos_str = f"{Colors.YELLOW}{reason if reason else 'N/A'}{Colors.RESET}"
                    
                    # --- FILTROS DE CALIDAD Y SEGURIDAD (Criterios de Inversión Real) ---
                    # 1. Recuperamos métricas clave
                    solvency = analysis.get("solvency", {})
                    profitability = analysis.get("profitability", {})
                    cash_flow = analysis.get("cash_flow", {})
                    market_data = analysis.get("market_data", {})
                    sector_group = analysis.get("sector_group") or normalize_sector(analysis.get("sector"))
                    debt_to_equity = solvency.get("Debt-to-Equity")
                    interest_coverage = solvency.get("Interest Coverage")
                    op_margin = profitability.get("Operating Margin")
                    gross_margin = profitability.get("Gross Margin")
                    roe = profitability.get("ROE")
                    roic = profitability.get("ROIC")
                    ebitda_growth = profitability.get("EBITDA Growth (3Y)")
                    pe_ratio = analysis.get("valuation", {}).get("P/E Ratio")
                    pb_ratio = analysis.get("valuation", {}).get("P/BV Ratio")
                    ev_to_ebitda = analysis.get("valuation", {}).get("EV/EBITDA")
                    dividend_yield = cash_flow.get("Dividend Yield")

                    # 2. Definimos los "Red Flags" (Banderas Rojas)
                    is_bankrupt_risk = (debt_to_equity is not None and debt_to_equity > 250) # Exceso de deuda
                    has_good_operating_margin = (op_margin is not None and op_margin >= 0.07) # Margen operativo >= 7%
                    #has_positive_ebitda_growth = (ebitda_growth is not None and ebitda_growth > 0) # EBITDA Growth > 0
                    is_fake_roe = (roe is not None and roe > 1.0)                          # ROE > 100% suele ser distorsión

                    has_pe_under_40 = (pe_ratio is not None and isinstance(pe_ratio, (int, float)) and pe_ratio <= 40)
                    has_pb_under_12 = (pb_ratio is not None and isinstance(pb_ratio, (int, float)) and pb_ratio < 1.2)
                    has_pb_under_15 = (pb_ratio is not None and isinstance(pb_ratio, (int, float)) and pb_ratio < 1.5)
                    has_dividend_over_4 = (dividend_yield is not None and isinstance(dividend_yield, (int, float)) and dividend_yield >= 0.04)
                    has_debt_ok_150 = (debt_to_equity is not None and isinstance(debt_to_equity, (int, float)) and debt_to_equity <= 150)
                    has_debt_ok_60 = (debt_to_equity is not None and isinstance(debt_to_equity, (int, float)) and debt_to_equity < 60)
                    has_interest_coverage = (interest_coverage is not None and isinstance(interest_coverage, (int, float)) and interest_coverage > 4)
                    has_ev_ebitda_under_8 = (ev_to_ebitda is not None and isinstance(ev_to_ebitda, (int, float)) and ev_to_ebitda < 8)
                    mos_good_default = isinstance(mos_normal, (float, int)) and mos_normal > 0.2
                    mos_good_looser = isinstance(mos_normal, (float, int)) and mos_normal > 0.15

                    if sector_group == "technology":
                        is_gem = (
                            mos_good_default and
                            moat >= 3 and
                            not is_fake_roe and
                            not is_bankrupt_risk and
                            op_margin is not None and op_margin > 0.15 and
                            ebitda_growth is not None and ebitda_growth > 0.10 and
                            has_pe_under_40
                        )
                    elif sector_group == "healthcare":
                        is_gem = (
                            mos_good_default and
                            moat >= 3 and
                            not is_fake_roe and
                            gross_margin is not None and gross_margin > 0.60 and
                            roic is not None and roic > 0.15
                        )
                    elif sector_group == "financials":
                        is_gem = (
                            mos_good_looser and
                            moat >= 3 and
                            not is_fake_roe and
                            has_pb_under_12 and
                            roe is not None and roe > 0.10
                        )
                    elif sector_group == "commodities":
                        fifty_two_week_high = market_data.get("Fifty Two Week High")
                        is_gem = (
                            mos_good_default and
                            moat >= 3 and
                            not is_fake_roe and
                            has_debt_ok_60 and
                            has_ev_ebitda_under_8 and
                            analysis.get("price") is not None and
                            fifty_two_week_high is not None and
                            analysis.get("price") < 0.8 * fifty_two_week_high
                        )
                    elif sector_group in {"utilities", "industrials"}:
                        is_gem = (
                            mos_good_looser and
                            moat >= 3 and
                            not is_fake_roe and
                            has_debt_ok_150 and
                            has_interest_coverage and
                            op_margin is not None and op_margin >= 0.07
                        )
                    elif sector_group == "reit":
                        is_gem = (
                            mos_good_looser and
                            moat >= 3 and
                            not is_fake_roe and
                            has_pb_under_15 and
                            has_dividend_over_4
                        )
                    elif sector_group == "staples":
                        is_gem = (
                            mos_good_default and
                            moat >= 4 and
                            not is_fake_roe and
                            has_good_operating_margin and
                            ebitda_growth is not None and ebitda_growth >= 0.04 and
                            not is_bankrupt_risk
                        )
                    elif sector_group == "consumer_cyclical":
                        is_gem = (
                            mos_good_default and
                            moat >= 3 and
                            not is_bankrupt_risk and
                            has_good_operating_margin and
                            not is_fake_roe and
                            pe_ratio is not None and isinstance(pe_ratio, (int, float)) and pe_ratio < 30 and
                            debt_to_equity is not None and isinstance(debt_to_equity, (int, float)) and debt_to_equity < 100
                        )
                    else:
                        is_gem = (
                            mos_good_default and 
                            moat >= 3 and 
                            not is_bankrupt_risk and 
                            has_good_operating_margin and 
                            #has_positive_ebitda_growth and 
                            not is_fake_roe and
                            pe_ratio is not None and isinstance(pe_ratio, (int, float)) and pe_ratio < 30 and
                            debt_to_equity is not None and isinstance(debt_to_equity, (int, float)) and debt_to_equity < 100
                        )

                    status_color = Colors.GREEN if is_gem else ""
                    
                    print(f"[{i}/{total_tickers}] {status_color}Analyzed: {ticker:<8} | Moat: {moat} | MOS: {mos_str}{Colors.RESET}")

                    # --- FILTRADO PARA GUARDAR ---
                    if is_gem:
                        undervalued_opportunities.append(analysis)
            except Exception as e:
                print(f"{Colors.RED}Error con {ticker}: {e}{Colors.RESET}")

    # --- GENERACIÓN DEL ARCHIVO FINAL ---
    if undervalued_opportunities:
        # Ordenamos por mayor Margen de Seguridad
        undervalued_opportunities.sort(key=lambda x: x['margin_of_safety']['Normal'], reverse=True)
        
        output_dir = ".\\infravaloradas"
        os.makedirs(output_dir, exist_ok=True)
        today_str = datetime.now().strftime("%Y-%m-%d")
        output_file = os.path.join(output_dir, f"infravaloradas_{today_str}.txt")
        
        with open(output_file, "w", encoding="utf-8") as f:
            # PARTE 1: TABLA RESUMEN (Vista rápida)
            f.write("=================================================================================\n")
            f.write(f"RESUMEN DE ACCIONES INFRAVALORADAS - {today_str}\n")
            f.write("=================================================================================\n")
            f.write(f"{'Ticker':<10} | {'Precio':<10} | {'V.I. Normal':<12} | {'Margen (MOS)':<12} | {'MOAT'}\n")
            f.write("-" * 81 + "\n")
            for op in undervalued_opportunities:
                f.write(f"{op['ticker']:<10} | ${op['price']:<9.2f} | ${op['intrinsic_value']['Normal']:<11.2f} | {op['margin_of_safety']['Normal']:<12.2%} | {op['moat_score']}/7\n")
            
            f.write("\n\n")

            # PARTE 2: ANÁLISIS DETALLADO (Copia de la terminal por cada empresa)
            f.write("=================================================================================\n")
            f.write("DETALLE EXTENDIDO DE CADA OPORTUNIDAD\n")
            f.write("=================================================================================\n")
            
            for analysis in undervalued_opportunities:
                f.write(f"\n\n{'#'*60}\n")
                f.write(f"### ANÁLISIS DE {analysis['ticker']}\n")
                f.write(f"{'#'*60}\n")
                f.write(f"Current Price: ${analysis['price']:.2f}\n")
                f.write(f"MOAT Score: {analysis['moat_score']}/7\n")
                f.write(f"Sector: {analysis.get('sector') or analysis.get('sector_group') or 'N/A'}\n")

                f.write("\n--- VALUATION ---\n")
                for key, value in analysis["valuation"].items():
                    f.write(f"{key}: {value:.2f}\n" if isinstance(value, (int, float)) else f"{key}: N/A\n")
                
                f.write("\n--- SOLVENCY & HEALTH ---\n")
                for key, value in analysis["solvency"].items():
                    if key == "Net Debt":
                        f.write(f"{key}: ${value:,.0f}\n")
                    else:
                        f.write(f"{key}: {value:.2f}\n" if isinstance(value, (int, float)) else f"{key}: N/A\n")

                f.write("\n--- PROFITABILITY & EFFICIENCY ---\n")
                for key, value in analysis["profitability"].items():
                    if key == "EBITDA" and isinstance(value, (int, float)):
                        f.write(f"{key}: ${value:,.0f}\n")
                    elif key == "EBITDA Growth (3Y)" and isinstance(value, (int, float)):
                        f.write(f"{key}: {value:.2%}\n")
                    else:
                        f.write(f"{key}: {value:.2%}\n" if isinstance(value, (int, float)) else f"{key}: N/A\n")

                f.write("\n--- CASH FLOW ---\n")
                for key, value in analysis["cash_flow"].items():
                    if "Free Cash Flow" in key and isinstance(value, (int, float)):
                        f.write(f"{key}: ${value:,.0f}\n")
                    elif "Yield" in key and isinstance(value, (int, float)):
                        val = value/100 if value > 1 else value
                        f.write(f"{key}: {val:.2%}\n")
                    else:
                        f.write(f"{key}: N/A\n")

                f.write("\n--- INTRINSIC VALUE & MARGIN OF SAFETY ---\n")
                for scenario in ["Ultra Pessimistic", "Pessimistic", "Normal", "Optimistic", "Ultra Optimistic"]:
                    iv = analysis['intrinsic_value'].get(scenario)
                    mos = analysis['margin_of_safety'].get(scenario)
                    # Verifica que AMBOS sean números (int o float) y no None
                    if isinstance(iv, (int, float)) and isinstance(mos, (int, float)):
                        f.write(f"{scenario:<18}: IV ${iv:>8.2f} | MOS {mos:>8.2%}\n")
                    else:
                        # Si alguno es None o un mensaje de error, escribe el texto tal cual
                        iv_text = f"${iv:.2f}" if isinstance(iv, (int, float)) else "N/A"
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