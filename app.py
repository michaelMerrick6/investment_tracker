#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict

import pandas as pd
import plotly.graph_objects as go


import streamlit as st

# ---------- Types & constants ----------
class Frequency(str, Enum):
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"
    quarterly = "quarterly"
    yearly = "yearly"

FREQ_TO_PERIODS: Dict[Frequency, int] = {
    Frequency.daily: 365,
    Frequency.weekly: 52,
    Frequency.monthly: 12,
    Frequency.quarterly: 4,
    Frequency.yearly: 1,
}

class AccountType(str, Enum):
    roth = "Roth IRA"
    k401 = "401(k)"
    brokerage = "Brokerage"


@dataclass
class ResultRow:
    period: int
    year: int
    user_contribution: float
    employer_match: float
    total_contributed: float   # cumulative user + employer
    balance_nominal: float     # end-of-period balance


# ---------- Core math ----------
def periodic_rate(annual_rate: float, periods_per_year: int) -> float:
    return (1 + annual_rate) ** (1 / periods_per_year) - 1


def simulate(
    contrib: float,
    freq: Frequency,
    annual_rate: float,
    years: int,
    deposit_at_start: bool,
    account_type: AccountType,
    employer_match_pct: float = 0.0,
    tax_drag_annual: float = 0.0,
) -> List[ResultRow]:

    ppy = FREQ_TO_PERIODS[freq]
    r = periodic_rate(annual_rate, ppy)
    tax_drag = periodic_rate(tax_drag_annual, ppy) if account_type == AccountType.brokerage else 0.0

    total_periods = years * ppy
    balance = 0.0
    total_user = 0.0
    total_match = 0.0
    rows: List[ResultRow] = []

    for p in range(1, total_periods + 1):
        match_amt = 0.0

        # Deposit at start (always True now)
        if deposit_at_start:
            if account_type == AccountType.k401:
                match_amt = contrib * employer_match_pct
            balance += contrib + match_amt
            total_user += contrib
            total_match += match_amt

        # Growth
        gain = balance * r
        if account_type == AccountType.brokerage and tax_drag > 0:
            tax = gain * (tax_drag / r) if r != 0 else 0
            balance += (gain - tax)
        else:
            balance += gain

        # (no end deposit path anymore)

        year = (p - 1) // ppy + 1
        rows.append(
            ResultRow(
                period=p,
                year=year,
                user_contribution=contrib,
                employer_match=match_amt,
                total_contributed=total_user + total_match,
                balance_nominal=balance,
            )
        )

    return rows


def to_yearly_df(rows: List[ResultRow], periods_per_year: int) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["Year","User Contributed","Employer Match (Yr)",
                                     "Total Contributed","Balance","Gain"])

    out = []
    last_year = rows[-1].year
    for yr in range(1, last_year + 1):
        start = (yr - 1) * periods_per_year
        end = yr * periods_per_year
        chunk = rows[start:end]

        user_yr = sum(r.user_contribution for r in chunk)
        match_yr = sum(r.employer_match for r in chunk)
        total_contrib_cum = chunk[-1].total_contributed
        balance = chunk[-1].balance_nominal
        gain = balance - total_contrib_cum

        out.append({
            "Year": yr,
            "User Contributed": user_yr,
            "Employer Match (Yr)": match_yr,
            "Total Contributed": total_contrib_cum,
            "Balance": balance,
            "Gain": gain,
        })
    return pd.DataFrame(out)


def add_inflation_columns(df: pd.DataFrame, inflation_rate: float) -> pd.DataFrame:
    if df.empty:
        return df
    if inflation_rate <= 0:
        df["Real Total Contributed"] = df["Total Contributed"]
        df["Real Balance"] = df["Balance"]
        df["Real Gain"] = df["Gain"]
        return df

    factors = (1 + inflation_rate) ** df["Year"]
    df["Real Total Contributed"] = df["Total Contributed"] / factors
    df["Real Balance"] = df["Balance"] / factors
    df["Real Gain"] = df["Real Balance"] - df["Real Total Contributed"]
    return df


def make_plot(df: pd.DataFrame) -> go.Figure:
    """Always show real series + nominal balance line."""
    fig = go.Figure()

    # Real contributed (light blue, dotted)
    fig.add_trace(go.Scatter(
        x=df["Year"], y=df["Real Total Contributed"],
        mode="lines+markers",
        name="Total Contributed (Real $)",
        line=dict(color="#74a9cf", width=2, dash="dot"),
        hovertemplate="%{y:$,.2f}<extra></extra>",
    ))

    # Real balance (dark blue)
    fig.add_trace(go.Scatter(
        x=df["Year"], y=df["Real Balance"],
        mode="lines+markers",
        name="Balance (Real $)",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="%{y:$,.2f}<extra></extra>",
    ))

    # Nominal balance (green)
    fig.add_trace(go.Scatter(
        x=df["Year"], y=df["Balance"],
        mode="lines",
        name="Balance (Nominal $)",
        line=dict(color="#28a745", width=2),
        hovertemplate="%{y:$,.2f}<extra></extra>",
    ))

    fig.update_layout(
        title="Portfolio Growth (Real $)",
        xaxis_title="Year",
        yaxis_title="USD",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


# ---------- helpers for money formatting ----------
def fmt_money_top(x: float) -> str:
    # abbreviate only at ≥ $10B
    return f"${x/1_000_000_000:,.2f}B" if abs(x) >= 10_000_000_000 else f"${x:,.2f}"

def fmt_money_bottom(x: float) -> str:
    # bottom row abbreviates at ≥ $1B
    return f"${x/1_000_000_000:,.2f}B" if abs(x) >= 1_000_000_000 else f"${x:,.2f}"


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Investment Growth Tracker", layout="centered")
st.markdown("""
<style>
div[data-testid="stMetricValue"]{
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: clip !important;
  word-break: break-word !important;
  font-size: 1.8rem !important;
  line-height: 1.15;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 Investment Growth Tracker")

col1, col2 = st.columns(2)
with col1:
    acct_str  = st.selectbox("Account type", [a.value for a in AccountType])
    freq_str  = st.selectbox("Contribution frequency", [f.value for f in Frequency], index=1)
    contrib   = st.number_input("Contribution per period ($)", min_value=0.0, step=10.0, value=100.0)
with col2:
    annual_rate_pct   = st.number_input("Avg annual growth rate (%)", value=8.0, step=0.5)
    inflation_rate_pct = st.number_input("Annual inflation rate (%)", value=2.5, step=0.1)
    years             = st.number_input("Number of years", min_value=1, step=1, value=30)

# Extra inputs per account
employer_match_pct = 0.0
tax_drag_pct = 0.0
acct = AccountType(acct_str)

if acct == AccountType.k401:
    match_pct_input = st.number_input(
        "Employer match (% of your contribution)", min_value=0.0, max_value=100.0, step=0.1, value=6.0
    )
    employer_match_pct = match_pct_input / 100.0
elif acct == AccountType.brokerage:
    tax_drag_pct = st.number_input(
        "Annual tax drag on gains (%)", min_value=0.0, max_value=100.0, step=0.5, value=15.0
    )

# Cast & compute
freq             = Frequency(freq_str)
annual_rate      = annual_rate_pct / 100.0
inflation_rate   = inflation_rate_pct / 100.0
tax_drag_annual  = tax_drag_pct / 100.0

rows = simulate(
    contrib=contrib,
    freq=freq,
    annual_rate=annual_rate,
    years=years,
    deposit_at_start=True,              # <— forced
    account_type=acct,
    employer_match_pct=employer_match_pct,
    tax_drag_annual=tax_drag_annual,
)

ppy = FREQ_TO_PERIODS[freq]
df_yearly = to_yearly_df(rows, ppy)
df_yearly = add_inflation_columns(df_yearly, inflation_rate)

if df_yearly.empty:
    st.warning("No data to display.")
    st.stop()

# ----- Metrics -----
final_nom = df_yearly["Balance"].iloc[-1]
total_nom = df_yearly["Total Contributed"].iloc[-1]
gain_nom  = final_nom - total_nom

final_real = df_yearly["Real Balance"].iloc[-1]
total_real = df_yearly["Real Total Contributed"].iloc[-1]
gain_real  = final_real - total_real
inflation_loss = final_nom - final_real

st.subheader("Summary")

# Top row (nominal)
col_fv, col_tc, col_tg = st.columns(3)
col_fv.metric("Final Value (Nominal)", fmt_money_top(final_nom))
col_tc.metric("Total Contributed",     fmt_money_top(total_nom))
col_tg.metric("Total Gain (Nominal)",  fmt_money_top(gain_nom))

# Bottom row (real / inflation)
block = """
<div style='text-align:left; font-size:1.2rem; line-height:1.35;'>
  <span style='color:{color_label};'>{label}</span><br>
  <b style='color:{color_val};font-size:1.4rem;'>{value}</b>
</div>
"""
with col_fv:
    st.markdown(block.format(
        color_label="#555", color_val="#555",
        label="Final Value (Real $)",
        value=fmt_money_bottom(final_real)
    ), unsafe_allow_html=True)

with col_tc:
    st.markdown(block.format(
        color_label="#555", color_val="#d9534f",
        label="Lost to Inflation",
        value=f"-{fmt_money_bottom(abs(inflation_loss))}"
    ), unsafe_allow_html=True)

with col_tg:
    sign = "+" if gain_real >= 0 else "-"
    st.markdown(block.format(
        color_label="#555", color_val="#28a745",
        label="Real Gain",
        value=f"{sign}{fmt_money_bottom(abs(gain_real))}"
    ), unsafe_allow_html=True)

# Chart
st.plotly_chart(make_plot(df_yearly), use_container_width=True)

# Table
with st.expander("Year-by-year table"):
    fmt = "${:,.2f}"
    st.dataframe(df_yearly.style.format({
        "User Contributed": fmt,
        "Employer Match (Yr)": fmt,
        "Total Contributed": fmt,
        "Balance": fmt,
        "Gain": fmt,
        "Real Total Contributed": fmt,
        "Real Balance": fmt,
        "Real Gain": fmt,
    }))

# Download
csv = df_yearly.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv,
                   file_name="investment_growth.csv", mime="text/csv")
