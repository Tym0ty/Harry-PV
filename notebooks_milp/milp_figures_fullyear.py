"""
Full-Year MILP — 7 Thesis Figures (spec §8.1)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path('../milp_outputs')

def main():
    solve = pd.read_csv(OUT / 'case_summary_fullyear.csv')
    replay = pd.read_csv(OUT / 'replay_summary_fullyear.csv')
    cases = solve['case'].values
    labels = ['C0\nDet PV\nDet Load', 'C1\nProb PV\nDet Load',
              'C2\nDet PV\nPert Load', 'C3\nProb PV\nPert Load']
    colors = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#FFA07A']

    fig_dir = OUT / 'figures_fullyear'
    fig_dir.mkdir(exist_ok=True)

    # ── Fig 1: Sizing comparison (CC, P_B, E_B) ─────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].bar(labels, solve['contract_kw'], color=colors)
    axes[0].set_ylabel('Contract Capacity (kW)')
    axes[0].set_title('Contract Capacity')
    for i, v in enumerate(solve['contract_kw']):
        axes[0].text(i, v + 20, f'{v:.0f}', ha='center', fontsize=9)

    axes[1].bar(labels, solve['bess_p_kw'], color=colors)
    axes[1].set_ylabel('BESS Power (kW)')
    axes[1].set_title('BESS Power')
    for i, v in enumerate(solve['bess_p_kw']):
        axes[1].text(i, v + 10, f'{v:.0f}', ha='center', fontsize=9)

    axes[2].bar(labels, solve['bess_e_kwh'], color=colors)
    axes[2].set_ylabel('BESS Energy (kWh)')
    axes[2].set_title('BESS Energy')
    for i, v in enumerate(solve['bess_e_kwh']):
        axes[2].text(i, v + 50, f'{v:.0f}', ha='center', fontsize=9)

    fig.suptitle('Figure 1: Sizing Comparison (CC*, P_B*, E_B*)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig1_sizing_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Fig 2: Replay annual cost comparison ─────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, replay['replay_total_M'], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Annual Cost (M NTD)')
    ax.set_title('Figure 2: Replay Annual Total Cost', fontsize=14, fontweight='bold')
    for i, v in enumerate(replay['replay_total_M']):
        ax.text(i, v + 0.1, f'{v:.2f}M', ha='center', fontsize=10)
    ax.set_ylim(bottom=94)
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig2_replay_cost.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Fig 3: Over-contract and worst-month bill ────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(labels, replay['replay_over_M'], color=colors)
    axes[0].set_ylabel('Over-Contract Fee (M NTD)')
    axes[0].set_title('Annual Over-Contract Fee')
    for i, v in enumerate(replay['replay_over_M']):
        axes[0].text(i, v + 0.005, f'{v:.2f}M', ha='center', fontsize=9)

    axes[1].bar(labels, replay['worst_bill_M'], color=colors)
    axes[1].set_ylabel('Worst Month Bill (M NTD)')
    axes[1].set_title('Worst-Month Bill')
    for i, v in enumerate(replay['worst_bill_M']):
        axes[1].text(i, v + 0.02, f'{v:.2f}M', ha='center', fontsize=9)

    fig.suptitle('Figure 3: Over-Contract & Worst-Month Bill (Replay)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig3_overcontract_worst.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Fig 4: RE20 achievement and T-REC ────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    re_vals = replay['RE_pct'].values
    trec_vals = replay['TREC_kWh'].values / 1e6  # MWh
    ax.bar(labels, re_vals, color=colors)
    ax.axhline(y=20, color='red', linestyle='--', linewidth=1.5, label='RE20 Target')
    ax.set_ylabel('RE Achievement (%)')
    ax.set_title('Figure 4: RE20 Achievement & T-REC Gap', fontsize=14, fontweight='bold')
    for i in range(len(cases)):
        ax.text(i, re_vals[i] + 0.2, f'{re_vals[i]:.1f}%\nT-REC: {trec_vals[i]:.1f} GWh',
                ha='center', fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig4_re20_trec.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Fig 5: Dispatch trace (C0 vs C1, sample week) ────────
    # Read solve dispatch from truth replay data
    truth = pd.read_parquet(OUT.parent / 'bridge_outputs_fullyear' / 'full_year_replay_truth_package.parquet')
    # Pick a summer week with high PV variability
    sample_days = truth[(truth['month_id'] == 7)]['day_index'].unique()[:7]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    for ax_idx, (case_label, pv_label) in enumerate([('C0', 'Det PV'), ('C1', 'Prob PV')]):
        ax = axes[ax_idx]
        subset = truth[truth['day_index'].isin(sample_days)].sort_values(['day_index', 'hour_local'])
        hours = np.arange(len(subset))
        ax.fill_between(hours, 0, subset['load_realized_kw'], alpha=0.3, color='gray', label='Load')
        ax.fill_between(hours, 0, subset['pv_realized_kw'], alpha=0.5, color='gold', label='PV')
        ax.set_ylabel('Power (kW)')
        ax.set_title(f'{case_label}: {pv_label} — July Week Dispatch')
        ax.legend(loc='upper right')
        ax.set_xlim(0, len(hours))

    axes[1].set_xlabel('Hour')
    fig.suptitle('Figure 5: Chronological Dispatch Trace (Truth)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig5_dispatch_trace.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Fig 6: Monthly bill breakdown (stacked) ──────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.2
    months = list(range(1, 13))
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    for ci, case_id in enumerate(cases):
        row = replay[replay['case_id'] == case_id].iloc[0]
        bills_str = str(row['monthly_bills'])
        # Parse using regex for robustness against numpy types
        import re
        pairs = re.findall(r'(\d+)\)?\s*:\s*([\d.]+)', bills_str)
        bills = {int(k): float(v) for k, v in pairs}
        vals = [bills.get(m, 0.0) for m in months]
        x = np.arange(len(months)) + ci * width
        ax.bar(x, vals, width, label=labels[ci].replace('\n', ' '), color=colors[ci])

    ax.set_xticks(np.arange(len(months)) + 1.5 * width)
    ax.set_xticklabels(month_labels)
    ax.set_ylabel('Monthly Bill (M NTD)')
    ax.set_title('Figure 6: Monthly Bill Breakdown (Replay)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig6_monthly_bills.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Fig 7: Design-to-replay gap comparison ───────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    gaps = replay['gap_pct'].values
    bar_colors = ['green' if g < 0 else 'red' for g in gaps]
    ax.bar(labels, gaps, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Gap (%)')
    ax.set_title('Figure 7: Design-to-Replay Gap', fontsize=14, fontweight='bold')
    for i, v in enumerate(gaps):
        ax.text(i, v + (0.1 if v >= 0 else -0.3), f'{v:+.1f}%', ha='center', fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig7_gap.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"All 7 figures saved to {fig_dir}")
    for f in sorted(fig_dir.glob('*.png')):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
