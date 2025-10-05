import matplotlib
matplotlib.use('Agg')

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# ============================================================================
# SIMPLE SINE WAVE MODEL - NO GROWTH, NO DIVIDENDS, PURE TIMING
# ============================================================================

def generate_simple_sine_wave(amplitude: float = 0.20) -> np.ndarray:
    """
    Generate one complete 8-year sine wave cycle.
    Year 0: neutral (100)
    Year 2: peak (100 + amplitude)
    Year 4: neutral (100)
    Year 6: trough (100 - amplitude)
    Year 8: neutral (100)
    """
    months = 8 * 12 + 1  # 8 years of months plus starting point
    time_years = np.arange(months) / 12
    
    # Pure sine wave, no drift
    prices = 100 * (1 + amplitude * np.sin(2 * np.pi * time_years / 8))
    
    return prices


def calculate_mark_to_market_tax(
    prices: np.ndarray,
    entry_month: int,
    investment_years: int,
    tax_rate: float
) -> Tuple[float, List[float], List[float]]:
    """
    Calculate mark-to-market taxes for a given entry point.
    
    Returns:
        total_tax: Total tax paid over investment period
        annual_taxes: Tax paid each year
        annual_losses: Loss carryforward each year
    """
    months = investment_years * 12
    entry_price = prices[entry_month]
    
    # Start with one share
    shares = 1.0
    tax_basis = entry_price
    loss_carryforward = 0.0
    
    total_tax_paid = 0.0
    annual_taxes = []
    annual_losses = []
    
    for year in range(investment_years):
        year_end_month = entry_month + (year + 1) * 12
        year_end_price = prices[year_end_month]
        
        current_value = shares * year_end_price
        unrealized_gain = current_value - tax_basis
        
        if unrealized_gain > 0:
            # Positive gain: tax it (after applying loss carryforward)
            taxable_gain = max(0, unrealized_gain - loss_carryforward)
            tax_due = taxable_gain * tax_rate
            
            annual_taxes.append(tax_due)
            total_tax_paid += tax_due
            
            # Update loss carryforward
            loss_carryforward = max(0, loss_carryforward - unrealized_gain)
            
            # Pay tax by selling shares
            if tax_due > 0:
                shares_to_sell = tax_due / year_end_price
                shares -= shares_to_sell
            
            # Step up basis
            tax_basis = shares * year_end_price
            annual_losses.append(loss_carryforward)
        else:
            # Loss: add to carryforward
            loss_amount = abs(unrealized_gain)
            loss_carryforward += loss_amount
            
            annual_taxes.append(0.0)
            annual_losses.append(loss_carryforward)
            
            # Step up basis to current value (mark-to-market)
            tax_basis = current_value
    
    return total_tax_paid, annual_taxes, annual_losses


def calculate_realization_tax(
    prices: np.ndarray,
    entry_month: int,
    investment_years: int,
    tax_rate: float
) -> float:
    """
    Calculate realization-based tax (only at exit).
    """
    entry_price = prices[entry_month]
    exit_month = entry_month + investment_years * 12
    exit_price = prices[exit_month]
    
    # One share bought at entry, sold at exit
    capital_gain = exit_price - entry_price
    
    if capital_gain > 0:
        return capital_gain * tax_rate
    else:
        return 0.0


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="Tax Timing Fairness",
    page_icon="⚖️",
    layout="wide"
)

st.markdown("""
    <style>
    .big-title {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">⚖️ The Tax Timing Unfairness</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Same Investment. Same Period. Same Outcome. Different Taxes?</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Settings")

amplitude = st.sidebar.slider(
    "Market Swing (±%)",
    min_value=10.0,
    max_value=30.0,
    value=20.0,
    step=5.0
) / 100

tax_rate = st.sidebar.slider(
    "Tax Rate (%)",
    min_value=10.0,
    max_value=40.0,
    value=20.0,
    step=5.0
) / 100

st.sidebar.markdown("---")
st.sidebar.markdown("""
### The Setup
- **Market Cycle**: 8 years
- **Investment Period**: 8 years
- **No growth trend**: Flat baseline
- **No dividends**: Pure price movement
- **5 Entry Points**: Years 0, 2, 4, 6, 8

### The Question
If all investors end with the **same final value**, 
should they pay **different taxes** based on 
when they entered?
""")

# Generate the sine wave
prices = generate_simple_sine_wave(amplitude)
time_years = np.arange(len(prices)) / 12

# Define entry points (years 0, 2, 4, 6, 8)
entry_years = [0, 2, 4, 6]
entry_months = [int(y * 12) for y in entry_years]
entry_labels = [f"Year {y}" for y in entry_years]

# Calculate taxes for each entry point
investment_years = 8

mtm_taxes = []
real_taxes = []
annual_tax_details = []

for entry_month in entry_months:
    mtm_tax, annual_taxes, annual_losses = calculate_mark_to_market_tax(
        prices, entry_month, investment_years, tax_rate
    )
    real_tax = calculate_realization_tax(
        prices, entry_month, investment_years, tax_rate
    )
    
    mtm_taxes.append(mtm_tax)
    real_taxes.append(real_tax)
    annual_tax_details.append((annual_taxes, annual_losses))

# Main visualization
st.header("📈 The Market Cycle")

fig1, ax1 = plt.subplots(figsize=(14, 7))

# Plot the sine wave
ax1.plot(time_years, prices, 'b-', linewidth=3, label='Market Price')

# Mark entry and exit points
colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
for i, (entry_month, label, color) in enumerate(zip(entry_months, entry_labels, colors)):
    entry_year = entry_month / 12
    exit_month = entry_month + investment_years * 12
    exit_year = exit_month / 12
    
    # Entry point (circle)
    ax1.scatter([entry_year], [prices[entry_month]], s=400, color=color,
               edgecolors='black', linewidth=3, zorder=5, marker='o',
               label=f'Entry at {label}')
    
    # Exit point (star)
    ax1.scatter([exit_year], [prices[exit_month]], s=600, color=color,
               edgecolors='black', linewidth=3, zorder=6, marker='*')
    
    # Connection line
    ax1.plot([entry_year, exit_year], [prices[entry_month], prices[exit_month]],
            color=color, linestyle='--', linewidth=2, alpha=0.6)

# Horizontal line at 100
ax1.axhline(y=100, color='gray', linestyle=':', linewidth=2, alpha=0.5)

ax1.set_xlabel('Years', fontsize=14, fontweight='bold')
ax1.set_ylabel('Price ($)', fontsize=14, fontweight='bold')
ax1.set_title('One Complete Market Cycle (8 Years) | All Investors Start & End at Same Price',
             fontsize=16, fontweight='bold')
ax1.legend(loc='upper right', fontsize=11, ncol=2)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(70, 130)

st.pyplot(fig1)

# Key insight box
st.info("""
🔍 **Key Insight**: All four entry points lead to exits at the SAME PRICE. 
Everyone's final portfolio value is identical. Everyone experienced the same market cycle.
But under mark-to-market taxation, they pay **wildly different taxes**!
""")

# Tax comparison
st.header("💰 Tax Bills: The Unfairness Revealed")

col1, col2 = st.columns(2)

with col1:
    st.subheader("❌ Mark-to-Market Taxation")
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    bars = ax2.bar(entry_labels, mtm_taxes, color=colors, 
                   edgecolor='black', linewidth=2, width=0.6)
    
    # Add value labels
    for bar, tax in zip(bars, mtm_taxes):
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'${tax:.2f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=12)
        else:
            ax2.text(bar.get_x() + bar.get_width()/2., 0.5,
                    '$0.00', ha='center', va='bottom',
                    fontweight='bold', fontsize=12)
    
    ax2.set_ylabel('Total Tax Paid ($)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Entry Point', fontsize=12, fontweight='bold')
    ax2.set_title('Tax Bills Under Mark-to-Market', fontsize=14, fontweight='bold', color='#e74c3c')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(mtm_taxes) * 1.3 if max(mtm_taxes) > 0 else 10)
    
    st.pyplot(fig2)
    
    # Show variation
    if max(mtm_taxes) > 0:
        tax_range = max(mtm_taxes) - min(mtm_taxes)
        st.error(f"""
        **Tax Variation: ${tax_range:.2f}**
        
        The highest tax is **{max(mtm_taxes)/min([t for t in mtm_taxes if t > 0] + [1]):.1f}x** 
        the lowest tax (excluding zeros)!
        """)

with col2:
    st.subheader("✅ Realization-Based Taxation")
    
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    
    bars = ax3.bar(entry_labels, real_taxes, color='#2ecc71',
                   edgecolor='black', linewidth=2, width=0.6)
    
    # Add value labels
    for bar, tax in zip(bars, real_taxes):
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'${tax:.2f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=12)
        else:
            ax3.text(bar.get_x() + bar.get_width()/2., 0.5,
                    '$0.00', ha='center', va='bottom',
                    fontweight='bold', fontsize=12)
    
    ax3.set_ylabel('Total Tax Paid ($)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Entry Point', fontsize=12, fontweight='bold')
    ax3.set_title('Tax Bills Under Realization', fontsize=14, fontweight='bold', color='#2ecc71')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, max(mtm_taxes) * 1.3 if max(mtm_taxes) > 0 else 10)
    
    st.pyplot(fig3)
    
    # Show fairness
    st.success(f"""
    **Tax Variation: ${max(real_taxes) - min(real_taxes):.2f}**
    
    All investors pay the SAME tax because they had the SAME actual gain!
    """)

# Detailed breakdown
st.header("📊 Year-by-Year Breakdown")

tab1, tab2, tab3, tab4 = st.tabs([f"Entry at {label}" for label in entry_labels])

for tab, label, entry_month, (annual_taxes, annual_losses), color in zip(
    [tab1, tab2, tab3, tab4], entry_labels, entry_months, annual_tax_details, colors
):
    with tab:
        st.subheader(f"Investor Entering at {label}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Year-by-year tax chart
            fig, ax = plt.subplots(figsize=(10, 5))
            
            years = np.arange(1, investment_years + 1)
            bars = ax.bar(years, annual_taxes, color=color, edgecolor='black', linewidth=1.5)
            
            for bar, tax in zip(bars, annual_taxes):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'${tax:.2f}', ha='center', va='bottom',
                           fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Year', fontsize=11, fontweight='bold')
            ax.set_ylabel('Tax Paid ($)', fontsize=11, fontweight='bold')
            ax.set_title(f'Annual Taxes Paid (Entry at {label})', fontsize=12, fontweight='bold')
            ax.set_xticks(years)
            ax.grid(True, alpha=0.3, axis='y')
            
            st.pyplot(fig)
        
        with col2:
            st.write("**Annual Tax Breakdown**")
            for year, tax, loss in zip(range(1, investment_years + 1), annual_taxes, annual_losses):
                if tax > 0:
                    st.write(f"Year {year}: 💸 ${tax:.2f}")
                else:
                    st.write(f"Year {year}: ✋ $0.00")
                if loss > 0:
                    st.caption(f"   Loss carryforward: ${loss:.2f}")
            
            st.markdown("---")
            st.metric("Total Tax", f"${sum(annual_taxes):.2f}")

# Bottom line
st.markdown("---")
st.header("🎯 The Bottom Line")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Entry Points Tested", len(entry_months))
    st.metric("Investment Period", "8 years")

with col2:
    st.metric("MTM Tax Range", f"${max(mtm_taxes) - min(mtm_taxes):.2f}")
    st.metric("Realization Tax Range", f"${max(real_taxes) - min(real_taxes):.2f}")

with col3:
    if max(mtm_taxes) > 0:
        unfairness_factor = max(mtm_taxes) / min([t for t in mtm_taxes if t > 0] + [1])
        st.metric("MTM Unfairness Factor", f"{unfairness_factor:.1f}x")
    st.metric("Realization Unfairness", "1.0x (Fair!)")

st.markdown("---")

st.error("""
### ❌ Mark-to-Market Taxation is Fundamentally Unfair

Two investors with:
- ✅ **Same initial investment**
- ✅ **Same market index**
- ✅ **Same investment period**
- ✅ **Same final portfolio value**
- ✅ **Same actual economic gain**

Pay **WILDLY DIFFERENT TAXES** based purely on **WHEN they entered the market**.

This is taxation of **LUCK**, not taxation of **GAINS**.
""")

st.success("""
### ✅ Realization-Based Taxation is Fair

All investors with the same actual gain pay the same tax, regardless of market timing.

**This is basic fairness. This is horizontal equity. This is justice.**
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; font-size: 0.8rem; color: #666;">
    <p><strong>Tax Timing Fairness Demonstrator</strong></p>
    <p>Simple. Clear. Devastating.</p>
</div>
""", unsafe_allow_html=True)
