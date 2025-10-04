import matplotlib
matplotlib.use('Agg')

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TaxSettings:
    """Tax configuration"""
    capital_gains_rate: float
    dividend_tax_rate: float
    dividend_yield: float = 0.0175  # Annual dividend yield


@dataclass
class MarketCycle:
    """Sine wave market model parameters"""
    initial_price: float = 100.0
    annual_drift: float = 0.07  # 7% annual trend growth
    amplitude: float = 0.20  # ±20% oscillation
    cycle_years: float = 8.0  # 8-year business cycle
    

class TaxLot:
    """Track cost basis for shares"""
    def __init__(self, shares: float, cost_basis: float, month: int):
        self.shares = shares
        self.cost_basis = cost_basis
        self.month = month


# ============================================================================
# MARKET MODEL
# ============================================================================

def generate_sine_wave_market(years: int, cycle_params: MarketCycle) -> np.ndarray:
    """
    Generate deterministic sine wave market prices.
    
    Price(t) = InitialPrice × (1 + drift)^t × (1 + amplitude × sin(2π × t / cycle_length))
    """
    months = years * 12
    time_in_years = np.arange(months + 1) / 12
    
    # Base exponential growth
    trend = cycle_params.initial_price * (1 + cycle_params.annual_drift) ** time_in_years
    
    # Cyclical component
    cycle_component = 1 + cycle_params.amplitude * np.sin(
        2 * np.pi * time_in_years / cycle_params.cycle_years
    )
    
    prices = trend * cycle_component
    
    return prices


# ============================================================================
# TAX SIMULATION
# ============================================================================

def simulate_mark_to_market(
    prices: np.ndarray,
    initial_investment: float,
    monthly_contribution: float,
    tax_settings: TaxSettings
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Simulate mark-to-market taxation.
    Returns: (portfolio_values, total_tax_paid, annual_taxes)
    """
    months = len(prices) - 1
    shares = initial_investment / prices[0]
    
    portfolio_values = np.zeros(months + 1)
    portfolio_values[0] = initial_investment
    
    tax_basis = initial_investment
    total_tax_paid = 0.0
    annual_taxes = []
    
    monthly_dividend_rate = tax_settings.dividend_yield / 12
    dividends_ytd = 0.0
    
    for month in range(months):
        price = prices[month + 1]
        
        # Dividends
        dividend = shares * price * monthly_dividend_rate
        dividends_ytd += dividend
        shares += dividend / price
        
        # Monthly contribution
        if monthly_contribution > 0:
            shares += monthly_contribution / price
        
        portfolio_values[month + 1] = shares * price
        
        # Annual tax event (every 12 months)
        if (month + 1) % 12 == 0:
            current_value = shares * price
            
            # Tax on dividends
            dividend_tax = dividends_ytd * tax_settings.dividend_tax_rate
            dividends_ytd = 0.0
            
            # Tax on unrealized gains (mark-to-market)
            unrealized_gain = current_value - tax_basis
            capital_gains_tax = max(0, unrealized_gain) * tax_settings.capital_gains_rate
            
            year_tax = dividend_tax + capital_gains_tax
            annual_taxes.append(year_tax)
            total_tax_paid += year_tax
            
            # Sell shares to pay tax
            if year_tax > 0:
                shares_to_sell = year_tax / price
                shares -= shares_to_sell
                portfolio_values[month + 1] = shares * price
            
            # Step up basis
            tax_basis = shares * price
    
    return portfolio_values, total_tax_paid, annual_taxes


def simulate_realization(
    prices: np.ndarray,
    initial_investment: float,
    monthly_contribution: float,
    tax_settings: TaxSettings
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Simulate realization-based taxation.
    Returns: (portfolio_values, total_tax_paid, annual_taxes)
    """
    months = len(prices) - 1
    shares = initial_investment / prices[0]
    
    portfolio_values = np.zeros(months + 1)
    portfolio_values[0] = initial_investment
    
    tax_lots: List[TaxLot] = [TaxLot(shares, prices[0], 0)]
    total_tax_paid = 0.0
    annual_taxes = []
    
    monthly_dividend_rate = tax_settings.dividend_yield / 12
    dividends_ytd = 0.0
    
    for month in range(months):
        price = prices[month + 1]
        
        # Dividends
        dividend = shares * price * monthly_dividend_rate
        dividends_ytd += dividend
        new_shares = dividend / price
        shares += new_shares
        tax_lots.append(TaxLot(new_shares, price, month))
        
        # Monthly contribution
        if monthly_contribution > 0:
            new_shares = monthly_contribution / price
            shares += new_shares
            tax_lots.append(TaxLot(new_shares, price, month))
        
        portfolio_values[month + 1] = shares * price
        
        # Annual tax on dividends only
        if (month + 1) % 12 == 0:
            dividend_tax = dividends_ytd * tax_settings.dividend_tax_rate
            dividends_ytd = 0.0
            
            annual_taxes.append(dividend_tax)
            total_tax_paid += dividend_tax
            
            # Sell shares to pay dividend tax
            if dividend_tax > 0:
                shares_to_sell = dividend_tax / price
                shares -= shares_to_sell
                # Remove from tax lots (FIFO)
                _remove_shares_fifo(tax_lots, shares_to_sell)
                portfolio_values[month + 1] = shares * price
    
    # Final liquidation - realize all capital gains
    final_price = prices[-1]
    total_gain = sum((lot.shares * final_price - lot.shares * lot.cost_basis) 
                     for lot in tax_lots)
    final_tax = max(0, total_gain) * tax_settings.capital_gains_rate
    
    # Also tax remaining dividends
    if dividends_ytd > 0:
        final_tax += dividends_ytd * tax_settings.dividend_tax_rate
    
    total_tax_paid += final_tax
    final_value = shares * final_price - final_tax
    portfolio_values[-1] = final_value
    
    return portfolio_values, total_tax_paid, annual_taxes


def _remove_shares_fifo(tax_lots: List[TaxLot], shares_to_remove: float):
    """Remove shares from tax lots using FIFO"""
    remaining = shares_to_remove
    lots_to_remove = []
    
    for i, lot in enumerate(tax_lots):
        if remaining <= 0:
            break
        
        if lot.shares <= remaining:
            remaining -= lot.shares
            lots_to_remove.append(i)
        else:
            lot.shares -= remaining
            remaining = 0
    
    for i in reversed(lots_to_remove):
        tax_lots.pop(i)


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="Market Timing Tax Impact",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📈 Market Timing Tax Impact Simulator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">How Your Entry Point Affects Your Tax Bill</p>', unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("⚙️ Configuration")

st.sidebar.subheader("💰 Investment Parameters")
initial_investment = st.sidebar.number_input(
    "Initial Investment ($)",
    min_value=1000,
    value=100000,
    step=10000
)

monthly_contribution = st.sidebar.number_input(
    "Monthly Contribution ($)",
    min_value=0,
    value=1000,
    step=100
)

investment_years = st.sidebar.slider(
    "Investment Period (Years)",
    min_value=10,
    max_value=30,
    value=20,
    step=1
)

st.sidebar.subheader("📊 Market Model")
annual_drift = st.sidebar.slider(
    "Annual Trend Growth (%)",
    min_value=3.0,
    max_value=12.0,
    value=7.0,
    step=0.5
) / 100

amplitude = st.sidebar.slider(
    "Market Oscillation Amplitude (±%)",
    min_value=10.0,
    max_value=40.0,
    value=20.0,
    step=5.0
) / 100

cycle_years = st.sidebar.slider(
    "Business Cycle Length (Years)",
    min_value=4.0,
    max_value=12.0,
    value=8.0,
    step=0.5
)

st.sidebar.subheader("💵 Tax Rates")
capital_gains_rate = st.sidebar.slider(
    "Capital Gains Tax Rate (%)",
    min_value=0.0,
    max_value=50.0,
    value=20.0,
    step=1.0
) / 100

dividend_tax_rate = st.sidebar.slider(
    "Dividend Tax Rate (%)",
    min_value=0.0,
    max_value=50.0,
    value=15.0,
    step=1.0
) / 100

dividend_yield = st.sidebar.slider(
    "Annual Dividend Yield (%)",
    min_value=0.0,
    max_value=5.0,
    value=1.75,
    step=0.25
) / 100

st.sidebar.subheader("🎯 Entry Points to Compare")
num_entry_points = st.sidebar.slider(
    "Number of Entry Points",
    min_value=2,
    max_value=6,
    value=4,
    step=1
)

run_button = st.sidebar.button("🚀 Run Simulation", type="primary")

# Main content
if run_button:
    # Create market model
    cycle_params = MarketCycle(
        initial_price=100.0,
        annual_drift=annual_drift,
        amplitude=amplitude,
        cycle_years=cycle_years
    )
    
    tax_settings = TaxSettings(
        capital_gains_rate=capital_gains_rate,
        dividend_tax_rate=dividend_tax_rate,
        dividend_yield=dividend_yield
    )
    
    # Generate full market cycle
    full_prices = generate_sine_wave_market(investment_years + 10, cycle_params)
    
    # Determine entry points (spread across one full cycle)
    entry_months = []
    cycle_months = int(cycle_years * 12)
    for i in range(num_entry_points):
        entry_month = int(i * cycle_months / num_entry_points)
        entry_months.append(entry_month)
    
    # Visualize the market and entry points
    st.header("📈 Market Model & Entry Points")
    
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    
    time_years = np.arange(len(full_prices)) / 12
    ax1.plot(time_years, full_prices, 'b-', linewidth=2, label='Market Price')
    
    # Mark entry points
    colors = plt.cm.Set1(np.linspace(0, 1, num_entry_points))
    for i, entry_month in enumerate(entry_months):
        entry_year = entry_month / 12
        ax1.axvline(x=entry_year, color=colors[i], linestyle='--', linewidth=2,
                   label=f'Entry Point {i+1} (Year {entry_year:.1f})')
        ax1.scatter([entry_year], [full_prices[entry_month]], color=colors[i], 
                   s=200, zorder=5, edgecolors='black', linewidth=2)
    
    ax1.set_xlabel('Years', fontsize=12)
    ax1.set_ylabel('Market Price ($)', fontsize=12)
    ax1.set_title(f'Deterministic Market Model: {annual_drift*100:.1f}% Annual Drift, ±{amplitude*100:.0f}% Oscillation, {cycle_years:.1f}yr Cycle',
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    st.pyplot(fig1)
    
    # Run simulations for each entry point
    results_mtm = []
    results_real = []
    
    with st.spinner('Running simulations for all entry points...'):
        for entry_month in entry_months:
            # Extract price series for this entry point
            months = investment_years * 12
            prices = full_prices[entry_month:entry_month + months + 1]
            
            # Normalize to start at initial investment
            adjustment_factor = cycle_params.initial_price / prices[0]
            prices = prices * adjustment_factor
            
            # Run both tax models
            mtm_values, mtm_tax, mtm_annual = simulate_mark_to_market(
                prices, initial_investment, monthly_contribution, tax_settings
            )
            
            real_values, real_tax, real_annual = simulate_realization(
                prices, initial_investment, monthly_contribution, tax_settings
            )
            
            results_mtm.append({
                'values': mtm_values,
                'total_tax': mtm_tax,
                'annual_taxes': mtm_annual,
                'final_value': mtm_values[-1]
            })
            
            results_real.append({
                'values': real_values,
                'total_tax': real_tax,
                'annual_taxes': real_annual,
                'final_value': real_values[-1]
            })
    
    st.success("✅ Simulations Complete!")
    
    # Create comparison tabs
    tab1, tab2, tab3 = st.tabs(["📊 Tax Comparison", "📈 Portfolio Values", "💡 Key Insights"])
    
    with tab1:
        st.header("Total Taxes Paid by Entry Point")
        
        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart comparison
        entry_labels = [f'Entry {i+1}\n(Yr {entry_months[i]/12:.1f})' 
                       for i in range(num_entry_points)]
        x = np.arange(num_entry_points)
        width = 0.35
        
        mtm_taxes = [r['total_tax'] for r in results_mtm]
        real_taxes = [r['total_tax'] for r in results_real]
        
        bars1 = ax2.bar(x - width/2, mtm_taxes, width, label='Mark-to-Market',
                       color='#DC2F02', edgecolor='black', linewidth=1.5)
        bars2 = ax2.bar(x + width/2, real_taxes, width, label='Realization',
                       color='#06A77D', edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars1, mtm_taxes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                    f'${val:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        for bar, val in zip(bars2, real_taxes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                    f'${val:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax2.set_ylabel('Total Taxes Paid ($)', fontsize=12, fontweight='bold')
        ax2.set_title('Total Tax Burden by Entry Point', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(entry_labels)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Tax difference chart
        tax_differences = [mtm - real for mtm, real in zip(mtm_taxes, real_taxes)]
        colors_diff = ['#DC2F02' if d > 0 else '#06A77D' for d in tax_differences]
        
        bars3 = ax3.bar(x, tax_differences, width=0.6, color=colors_diff,
                       edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars3, tax_differences):
            height = bar.get_height()
            va = 'bottom' if height > 0 else 'top'
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'${val:,.0f}', ha='center', va=va, fontsize=9, fontweight='bold')
        
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.set_ylabel('Tax Difference (MTM - Realization) ($)', fontsize=12, fontweight='bold')
        ax3.set_title('Extra Tax from Mark-to-Market', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(entry_labels)
        ax3.grid(True, alpha=0.3, axis='y')
        
        st.pyplot(fig2)
        
        # Summary table
        st.subheader("Detailed Tax Comparison")
        
        comparison_data = {
            "Entry Point": entry_labels,
            "Mark-to-Market Tax": [f"${t:,.0f}" for t in mtm_taxes],
            "Realization Tax": [f"${t:,.0f}" for t in real_taxes],
            "Difference": [f"${d:,.0f}" for d in tax_differences],
            "MTM Premium (%)": [f"{(mtm/real - 1)*100:.1f}%" if real > 0 else "N/A" 
                               for mtm, real in zip(mtm_taxes, real_taxes)]
        }
        
        st.table(comparison_data)
    
    with tab2:
        st.header("Final Portfolio Values by Entry Point")
        
        fig3, (ax4, ax5) = plt.subplots(1, 2, figsize=(16, 6))
        
        mtm_finals = [r['final_value'] for r in results_mtm]
        real_finals = [r['final_value'] for r in results_real]
        
        # Portfolio value comparison
        bars4 = ax4.bar(x - width/2, mtm_finals, width, label='Mark-to-Market',
                       color='#FF6B6B', edgecolor='black', linewidth=1.5)
        bars5 = ax4.bar(x + width/2, real_finals, width, label='Realization',
                       color='#06A77D', edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars4, mtm_finals):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'${val/1e6:.2f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        for bar, val in zip(bars5, real_finals):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'${val/1e6:.2f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax4.set_ylabel('Final Portfolio Value ($)', fontsize=12, fontweight='bold')
        ax4.set_title('After-Tax Portfolio Values', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(entry_labels)
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Value difference
        value_differences = [real - mtm for mtm, real in zip(mtm_finals, real_finals)]
        colors_val = ['#06A77D' if d > 0 else '#DC2F02' for d in value_differences]
        
        bars6 = ax5.bar(x, value_differences, width=0.6, color=colors_val,
                       edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars6, value_differences):
            height = bar.get_height()
            va = 'bottom' if height > 0 else 'top'
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'${val:,.0f}', ha='center', va=va, fontsize=9, fontweight='bold')
        
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax5.set_ylabel('Value Difference (Real - MTM) ($)', fontsize=12, fontweight='bold')
        ax5.set_title('Extra Wealth Kept with Realization', fontsize=14, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(entry_labels)
        ax5.grid(True, alpha=0.3, axis='y')
        
        st.pyplot(fig3)
    
    with tab3:
        st.header("💡 Key Insights")
        
        max_mtm_tax = max(mtm_taxes)
        min_mtm_tax = min(mtm_taxes)
        max_real_tax = max(real_taxes)
        min_real_tax = min(real_taxes)
        
        avg_mtm = np.mean(mtm_taxes)
        avg_real = np.mean(real_taxes)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔴 Mark-to-Market: Unfair Timing Penalty")
            st.error(f"""
            **Tax Varies Wildly by Entry Timing:**
            - Highest Tax: ${max_mtm_tax:,.0f}
            - Lowest Tax: ${min_mtm_tax:,.0f}
            - **Range: ${max_mtm_tax - min_mtm_tax:,.0f}**
            - Variation: {(max_mtm_tax/min_mtm_tax - 1)*100:.1f}% difference
            
            Two investors with identical contributions and holding periods 
            pay vastly different taxes based purely on when they started!
            """)
        
        with col2:
            st.subheader("🟢 Realization: Fair and Consistent")
            st.success(f"""
            **Tax Based on Actual Gains, Not Timing:**
            - Highest Tax: ${max_real_tax:,.0f}
            - Lowest Tax: ${min_real_tax:,.0f}
            - **Range: ${max_real_tax - min_real_tax:,.0f}**
            - Variation: {(max_real_tax/min_real_tax - 1)*100:.1f}% difference
            
            Realization taxation is much more consistent - you pay tax 
            on your actual gains, not on arbitrary annual valuations.
            """)
        
        st.markdown("---")
        
        st.subheader("📊 The Bottom Line")
        
        total_diff = sum(tax_differences)
        avg_diff = np.mean(tax_differences)
        
        st.info(f"""
        **Mark-to-Market Costs More AND is Less Fair:**
        
        - **Average Extra Tax with MTM**: ${avg_diff:,.0f} per investor
        - **Average MTM Tax**: ${avg_mtm:,.0f}
        - **Average Realization Tax**: ${avg_real:,.0f}
        - **MTM Premium**: {(avg_mtm/avg_real - 1)*100:.1f}% higher
        
        **Political Message:**
        
        Mark-to-Market taxation punishes investors based on **luck** (when they entered the market), 
        not performance. Two people with identical investment strategies can pay vastly different taxes.
        
        Realization-based taxation is **fair** - you pay tax on actual realized gains, regardless of 
        market timing. Everyone with the same gains pays similar taxes.
        """)
        
        st.markdown("---")
        
        st.subheader("🎯 Advocacy Talking Points")
        
        st.markdown(f"""
        1. **"Mark-to-Market Punishes Market Timing Luck"**
           - Same investment, same period, but tax varies by ${max_mtm_tax - min_mtm_tax:,.0f} based on entry point
           - This is fundamentally unfair
        
        2. **"Realization Taxes Real Gains, Not Paper Fluctuations"**
           - Your tax bill reflects your actual profit, not arbitrary annual snapshots
           - Fair and predictable
        
        3. **"Mark-to-Market Costs Investors More"**
           - Average {(avg_mtm/avg_real - 1)*100:.1f}% higher tax burden
           - Same final value, more tax = less wealth for citizens
        
        4. **"Who Benefits from Mark-to-Market? Only Short-Term Politicians"**
           - Governments get cash flow today
           - Investors lose wealth
           - Economy suffers from reduced capital
        """)

else:
    st.info("👈 Configure parameters in the sidebar and click 'Run Simulation' to begin")
    
    st.markdown("""
    ## About This Tool
    
    This simulator demonstrates how **market entry timing** affects taxation under two models:
    
    ### Market Model:
    - **Deterministic sine wave** with upward trend
    - Represents business cycles (ups and downs) with long-term growth
    - Configurable drift rate, amplitude, and cycle length
    
    ### Tax Models:
    
    **Mark-to-Market:**
    - Taxes unrealized gains annually
    - Forces investors to sell shares to pay taxes
    - **Result: Tax bill varies wildly based on when you enter the market**
    
    **Realization:**
    - Taxes only realized gains (when you sell)
    - Final liquidation event at end of period
    - **Result: Fair taxation based on actual gains, not timing luck**
    
    ### Key Insight:
    
    Two investors with identical:
    - Investment amounts
    - Contribution schedules  
    - Holding periods
    - Final portfolio values
    
    Can pay **vastly different taxes** under mark-to-market, but **similar taxes** under realization.
    
    **This demonstrates fundamental unfairness in mark-to-market taxation.**
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; font-size: 0.8rem; color: #666;">
    <p><strong>Market Timing Tax Impact Simulator</strong></p>
    <p>Demonstrating Tax Policy Fairness</p>
</div>
""", unsafe_allow_html=True)
