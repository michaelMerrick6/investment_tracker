#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Literal, List, Dict

Frequency = Literal["weekly", "monthly", "quarterly", "yearly"]

FREQ_TO_PERIODS: Dict[Frequency, int] = {
    "weekly": 52,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1,
}

@dataclass
class ResultRow:
    period: int
    year: int
    contribution: float
    total_contributed: float
    balance: float

def periodic_rate(annual_rate: float, periods_per_year: int) -> float:
    """Convert nominal annual growth (e.g. 0.08) to per-period compound rate."""
    return (1 + annual_rate) ** (1 / periods_per_year) - 1

import pandas as pd

def to_yearly_df(rows, periods_per_year):
    # keep last row of each year
    yearly = [r for i, r in enumerate(rows) if (i + 1) % periods_per_year == 0]
    return pd.DataFrame({
        "Year": [r.year for r in yearly],
        "Total Contributed": [r.total_contributed for r in yearly],
        "Balance": [r.balance for r in yearly],
        "Gain": [r.balance - r.total_contributed for r in yearly],
    })

import matplotlib.pyplot as plt

def plot_matplotlib(df):
    plt.figure(figsize=(8, 5))
    plt.plot(df["Year"], df["Balance"], label="Balance")
    plt.plot(df["Year"], df["Total Contributed"], label="Total Contributed")
    plt.xlabel("Year")
    plt.ylabel("USD")
    plt.title("Portfolio Growth")
    plt.legend()
    plt.tight_layout()
    plt.show()
import plotly.graph_objects as go

def plot_plotly(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Year"], y=df["Balance"],
                             mode="lines", name="Balance"))
    fig.add_trace(go.Scatter(x=df["Year"], y=df["Total Contributed"],
                             mode="lines", name="Total Contributed"))
    fig.update_layout(
        title="Portfolio Growth",
        xaxis_title="Year",
        yaxis_title="USD",
        hovermode="x unified"
    )
    fig.show()

def simulate(
    contrib: float,
    freq: Frequency,
    annual_rate: float,
    years: int,
    deposit_at_start: bool = False
) -> List[ResultRow]:
    ppy = FREQ_TO_PERIODS[freq]
    r = periodic_rate(annual_rate, ppy)
    total_periods = years * ppy

    balance = 0.0
    total_contrib = 0.0
    rows: List[ResultRow] = []

    for p in range(1, total_periods + 1):
        # Optionally deposit at the beginning of the period
        if deposit_at_start:
            balance += contrib
            total_contrib += contrib

        # Grow for one period
        balance *= (1 + r)

        # Deposit at end of period (default)
        if not deposit_at_start:
            balance += contrib
            total_contrib += contrib

        year = (p - 1) // ppy + 1
        rows.append(ResultRow(p, year, contrib, total_contrib, balance))

    return rows

def main():
    print("=== Investment Growth Tracker ===")
    contrib = float(input("Contribution amount each period (e.g. 100): "))
    freq = input("Frequency (weekly/monthly/quarterly/yearly): ").strip().lower()
    if freq not in FREQ_TO_PERIODS:
        raise ValueError("Invalid frequency.")
    annual_rate = float(input("Average annual growth rate % (e.g. 8): ")) / 100.0
    years = int(input("Number of years (e.g. 30): "))
    deposit_timing = input("Deposit at start of period? (y/N): ").strip().lower()
    deposit_at_start = deposit_timing == "y"

    rows = simulate(contrib, freq, annual_rate, years, deposit_at_start)

    final_balance = rows[-1].balance
    total_contrib = rows[-1].total_contributed
    gain = final_balance - total_contrib

    print("\n--- Summary ---")
    print(f"Total contributed: ${total_contrib:,.2f}")
    print(f"Final value:       ${final_balance:,.2f}")
    print(f"Total gain:        ${gain:,.2f}")

    use_plot = input("Plot results? (none/matplotlib/plotly): ").strip().lower()
    if use_plot in {"matplotlib", "plotly"}:
        ppy = FREQ_TO_PERIODS[freq]
        df_yearly = to_yearly_df(rows, ppy)
        if use_plot == "matplotlib":
            plot_matplotlib(df_yearly)
        else:
            plot_plotly(df_yearly)
    show_table = input("Show year-by-year summary? (y/N): ").strip().lower()
    if show_table == "y":
        # collapse to last row of each year
        ppy = FREQ_TO_PERIODS[freq]
        yearly_rows = [r for idx, r in enumerate(rows) if (idx + 1) % ppy == 0]
        print("\nYear | Contributed | Balance")
        print("-----------------------------")
        for r in yearly_rows:
            print(f"{r.year:4d} | ${r.total_contributed:10,.2f} | ${r.balance:10,.2f}")

if __name__ == "__main__":
    main()
