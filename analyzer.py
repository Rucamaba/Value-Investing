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

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'

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

def score_moat(info, financials, cashflow, balance_sheet, total_debt, cash):
    """
    Scores the company's competitive moat based on profitability and consistency.
    """
    score = 0
    
    # 1. High Profitability (ROIC > 15%)
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

def analyze_ticker(ticker_symbol):
    data = get_financial_data(ticker_symbol)
    if not data:
        print(f"{Colors.YELLOW}Warning: Could not retrieve sufficient financial data for {ticker_symbol}. Skipping...{Colors.RESET}")
        return None
    
    info = data["info"]
    financials = data["financials"]
    balance_sheet = data["balance_sheet"]
    cashflow = data["cashflow"]

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
        roic = (ebit * 0.75) / invested_capital if invested_capital > 0 else None
    except:
        roic = info.get("returnOnInvestedCapital") # Fallback al dato de Yahoo

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

    # --- Cálculo del Valor Intrínseco ---
    error_reason = None
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
            current_price=current_price
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
    moat_score = score_moat(info, financials, cashflow, balance_sheet, total_debt, cash)

    peer_metrics = extract_peer_metrics_with_balance(
        info=info,
        financials=financials,
        balance_sheet=balance_sheet,
        cashflow=cashflow,
        fcf=fcf,
        debt_to_equity=debt_to_equity,
        operating_margin=operating_margin,
        roe=roe,
    )

    return {
        "ticker": ticker_symbol,
        "price": current_price,
        "error_reason": error_reason,
        "moat_score": moat_score,
        "valuation": {
            "P/E Ratio": pe_ratio,
            "P/BV Ratio": pb_ratio,
            "EV/EBITDA": ev_to_ebitda,
            "Price/FCF": peer_metrics.get("Price/FCF"),
        },
        "solvency": {
            "Debt-to-Equity": debt_to_equity,
            "Current Ratio": current_ratio,
            "Net Debt": total_debt - cash,
        },
        "profitability": {
            "ROE": roe,
            "ROE (multi-year avg)": peer_metrics.get("ROE (multi-year avg)"),
            "ROIC": roic,
            "Gross Margin": gross_margin,
            "Operating Margin": operating_margin,
            "Operating Margin (multi-year avg)": peer_metrics.get("Operating Margin (multi-year avg)"),
        },
        "cash_flow": {
            "Free Cash Flow (3Y Avg)": fcf,
            "Dividend Yield": dividend_yield,
        },
        "peer_metrics": peer_metrics,
        "peers": None,  # filled after industry benchmarks are available
        "intrinsic_value": intrinsic_values,
        "margin_of_safety": margin_of_safety,
        "company_name": info.get("longName"),
        "industry": peer_metrics.get("industry"),
        "sector": peer_metrics.get("sector"),
    }


def is_quality_gem(analysis: dict) -> bool:
    """
    Core gem definition: MOS + moat + red-flag filters + peer robustness.
    Peer data only blocks clear relative traps (worse vs industry on all checks).
    """
    mos_normal = analysis.get("margin_of_safety", {}).get("Normal")
    moat = analysis.get("moat_score")
    solvency = analysis.get("solvency", {})
    profitability = analysis.get("profitability", {})
    debt_to_equity = solvency.get("Debt-to-Equity")
    op_margin = profitability.get("Operating Margin")
    roe = profitability.get("ROE")

    is_bankrupt_risk = debt_to_equity is not None and debt_to_equity > 250
    is_losing_money = op_margin is not None and op_margin < 0.02
    is_fake_roe = roe is not None and roe > 1.0

    return (
        isinstance(mos_normal, float)
        and mos_normal > 0.2
        and moat is not None
        and moat >= 3
        and not is_bankrupt_risk
        and not is_losing_money
        and not is_fake_roe
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

        lines = []
        lines.append("=" * 80)
        lines.append(f"ANALISIS DE {analysis.get('company_name', ticker)} - {ticker}")
        lines.append("=" * 80)
        if isinstance(analysis.get("price"), (int, float)):
            lines.append(f"Precio actual: ${analysis['price']:.2f}")
        else:
            lines.append("Precio actual: N/A")
        lines.append(f"MOAT Score: {analysis.get('moat_score', 'N/A')}/7")
        lines.append(
            f"Industry: {analysis.get('industry', 'N/A')} | Sector: {analysis.get('sector', 'N/A')}"
        )
        lines.append("")

        lines.append("--- VALUATION ---")
        for key, value in (analysis.get("valuation") or {}).items():
            if isinstance(value, (int, float)):
                if "Price/FCF" in key:
                    lines.append(f"{key}: {value:.1f}x")
                else:
                    lines.append(f"{key}: {value:.2f}")
            else:
                lines.append(f"{key}: N/A")

        lines.append("")
        lines.append("--- SOLVENCY & HEALTH ---")
        for key, value in (analysis.get("solvency") or {}).items():
            if key == "Net Debt" and isinstance(value, (int, float)):
                lines.append(f"{key}: ${value:,.0f}")
            else:
                lines.append(
                    f"{key}: {value:.2f}" if isinstance(value, (int, float)) else f"{key}: N/A"
                )

        lines.append("")
        lines.append("--- PROFITABILITY & EFFICIENCY ---")
        for key, value in (analysis.get("profitability") or {}).items():
            lines.append(
                f"{key}: {value:.2%}" if isinstance(value, (int, float)) else f"{key}: N/A"
            )

        lines.append("")
        lines.append("--- CASH FLOW ---")
        for key, value in (analysis.get("cash_flow") or {}).items():
            if "Free Cash Flow" in key and isinstance(value, (int, float)):
                lines.append(f"{key}: ${value:,.0f}")
            elif "Yield" in key and isinstance(value, (int, float)):
                val = value / 100 if value > 1 else value
                lines.append(f"{key}: {val:.2%}")
            else:
                lines.append(f"{key}: N/A")

        lines.append("")
        lines.append("--- PEERS / INDUSTRIA ---")
        lines.extend(format_peer_section(analysis.get("peers")))

        lines.append("")
        lines.append("--- INTRINSIC VALUE & MARGIN OF SAFETY ---")
        for scenario in [
            "Ultra Pessimistic",
            "Pessimistic",
            "Normal",
            "Optimistic",
            "Ultra Optimistic",
        ]:
            iv = analysis.get("intrinsic_value", {}).get(scenario)
            mos = analysis.get("margin_of_safety", {}).get(scenario)
            if isinstance(iv, (int, float)) and isinstance(mos, (int, float)):
                lines.append(f"{scenario:<18}: IV ${iv:>8.2f} | MOS {mos:>8.2%}")
            else:
                iv_text = f"${iv:.2f}" if isinstance(iv, (int, float)) else "N/A"
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

    print("\n" + "=" * 70)
    company_name = analysis.get('company_name', analysis['ticker'])
    print(f"ANALISIS INDIVIDUAL: {company_name} ({analysis['ticker']})")
    print("=" * 70)

    if isinstance(analysis.get("price"), (int, float)):
        print(f"Precio actual: ${analysis['price']:.2f}")
    else:
        print("Precio actual: N/A")

    print(f"MOAT Score: {analysis.get('moat_score', 'N/A')}/7")
    if analysis.get("industry"):
        print(f"Industry: {analysis.get('industry')} | Sector: {analysis.get('sector', 'N/A')}")

    print("\n--- VALOR INTRINSECO Y MARGEN DE SEGURIDAD ---")
    for scenario in ["Ultra Pessimistic", "Pessimistic", "Normal", "Optimistic", "Ultra Optimistic"]:
        iv = analysis.get("intrinsic_value", {}).get(scenario)
        mos = analysis.get("margin_of_safety", {}).get(scenario)
        iv_text = f"${iv:.2f}" if isinstance(iv, (int, float)) else str(iv)
        mos_text = f"{mos:.2%}" if isinstance(mos, (int, float)) else "N/A"
        print(f"{scenario:<18}: IV {iv_text:>12} | MOS {mos_text:>8}")

    print("\n--- PEERS / INDUSTRIA ---")
    for line in format_peer_section(analysis.get("peers")):
        print(line)

    p_fcf = (analysis.get("valuation") or {}).get("Price/FCF")
    if isinstance(p_fcf, (int, float)):
        print(f"Price/FCF (company): {p_fcf:.1f}x")

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
                f.write(
                    f"{op['ticker']:<10} | ${op['price']:<9.2f} | "
                    f"${op['intrinsic_value']['Normal']:<11.2f} | "
                    f"{op['margin_of_safety']['Normal']:<12.2%} | "
                    f"{op['moat_score']}/7   | {edge:<8} | {industry}\n"
                )
            
            f.write("\n\n")

            f.write("=================================================================================\n")
            f.write("DETALLE EXTENDIDO DE CADA OPORTUNIDAD\n")
            f.write("=================================================================================\n")
            
            for analysis in undervalued_opportunities:
                f.write(f"\n\n{'#'*60}\n")
                f.write(f"### ANÁLISIS DE {analysis['ticker']}\n")
                f.write(f"{'#'*60}\n")
                f.write(f"Current Price: ${analysis['price']:.2f}\n")
                f.write(f"MOAT Score: {analysis['moat_score']}/7\n")
                f.write(f"Industry: {analysis.get('industry', 'N/A')} | Sector: {analysis.get('sector', 'N/A')}\n")

                f.write("\n--- VALUATION ---\n")
                for key, value in analysis["valuation"].items():
                    if isinstance(value, (int, float)):
                        if "Price/FCF" in key:
                            f.write(f"{key}: {value:.1f}x\n")
                        else:
                            f.write(f"{key}: {value:.2f}\n")
                    else:
                        f.write(f"{key}: N/A\n")
                
                f.write("\n--- SOLVENCY & HEALTH ---\n")
                for key, value in analysis["solvency"].items():
                    if key == "Net Debt":
                        f.write(f"{key}: ${value:,.0f}\n")
                    else:
                        f.write(f"{key}: {value:.2f}\n" if isinstance(value, (int, float)) else f"{key}: N/A\n")

                f.write("\n--- PROFITABILITY & EFFICIENCY ---\n")
                for key, value in analysis["profitability"].items():
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

                f.write("\n--- PEERS / INDUSTRIA ---\n")
                for line in format_peer_section(analysis.get("peers")):
                    f.write(line + "\n")

                f.write("\n--- INTRINSIC VALUE & MARGIN OF SAFETY ---\n")
                for scenario in ["Ultra Pessimistic", "Pessimistic", "Normal", "Optimistic", "Ultra Optimistic"]:
                    iv = analysis['intrinsic_value'].get(scenario)
                    mos = analysis['margin_of_safety'].get(scenario)
                    if isinstance(iv, (int, float)) and isinstance(mos, (int, float)):
                        f.write(f"{scenario:<18}: IV ${iv:>8.2f} | MOS {mos:>8.2%}\n")
                    else:
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