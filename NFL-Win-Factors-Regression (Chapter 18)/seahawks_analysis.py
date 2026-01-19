###########################################################################
### Seahawks Performance Analysis (2014-2017)                           ###
# Based on Chapter 18: What Makes NFL Teams Win?                          #
###########################################################################

import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import os

# Load data
nfldata = pd.read_csv("NFLdata.csv")

# Create outputs folder if it doesn't exist
outputs_dir = "outputs"
os.makedirs(outputs_dir, exist_ok=True)

# Create output file (overwrites previous)
output_file = os.path.join(outputs_dir, "seahawks_output.txt")

# Redirect output to both console and file
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

f = open(output_file, 'w')
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, f)

print("="*80)
print("SEATTLE SEAHAWKS PERFORMANCE ANALYSIS (2014-2017)")
print("="*80)
print()

# Train model on all data
model = ols('Margin~Q("RET TD")+PENDIF+Q("PY/A")+Q("DPY/A")+Q("RY/A")+Q("DRY/A")+TO+DTO-1',data = nfldata).fit()

# Extract Seahawks data
seahawks = nfldata[nfldata['Team'] == 'Seattle'].copy()
seahawks = seahawks.sort_values('Year')

# Make predictions
seahawks['Predicted_Margin'] = model.predict(seahawks)
seahawks['Residual'] = seahawks['Margin'] - seahawks['Predicted_Margin']

print("\n" + "="*80)
print("SEAHAWKS YEAR-BY-YEAR PERFORMANCE")
print("="*80)
for _, row in seahawks.iterrows():
    print(f"\n{int(row['Year'])} Season:")
    print(f"  Actual Margin:    {row['Margin']:>6.1f} points")
    print(f"  Predicted Margin: {row['Predicted_Margin']:>6.1f} points")
    print(f"  Difference:       {row['Residual']:>6.1f} points ({'overperformed' if row['Residual'] > 0 else 'underperformed'})")
    print(f"\n  Key Stats:")
    print(f"    Return TDs:     {int(row['RET TD'])}")
    print(f"    Penalty Diff:   {int(row['PENDIF'])}")
    print(f"    Pass Yards/Att: {row['PY/A']:.1f} (Def: {row['DPY/A']:.1f})")
    print(f"    Rush Yards/Att: {row['RY/A']:.1f} (Def: {row['DRY/A']:.1f})")
    print(f"    Turnovers:      {int(row['TO'])} (Forced: {int(row['DTO'])})")

print("\n" + "="*80)
print("SEAHAWKS 4-YEAR AVERAGES (2014-2017)")
print("="*80)
print(f"Average Margin:       {seahawks['Margin'].mean():>6.1f}")
print(f"Average Pass Yards/A: {seahawks['PY/A'].mean():>6.2f}")
print(f"Average Def Pass Y/A: {seahawks['DPY/A'].mean():>6.2f}")
print(f"Average Rush Yards/A: {seahawks['RY/A'].mean():>6.2f}")
print(f"Average Def Rush Y/A: {seahawks['DRY/A'].mean():>6.2f}")
print(f"Total Return TDs:     {int(seahawks['RET TD'].sum())}")

print("\n" + "="*80)
print("COMPARISON TO LEAGUE")
print("="*80)
print(f"Seahawks Avg Margin:  {seahawks['Margin'].mean():>6.1f}")
print(f"League Avg Margin:    {nfldata['Margin'].mean():>6.1f}")
print(f"Seahawks vs League:   {seahawks['Margin'].mean() - nfldata['Margin'].mean():>+6.1f} points better")

print("\n" + "="*80)
print("MODEL COEFFICIENTS (What Matters Most)")
print("="*80)
print(f"Pass Yards/Attempt:   {model.params['Q(\"PY/A\")']:>+7.2f}")
print(f"Def Pass Yards/Att:   {model.params['Q(\"DPY/A\")']:>+7.2f}")
print(f"Rush Yards/Attempt:   {model.params['Q(\"RY/A\")']:>+7.2f}")
print(f"Def Rush Yards/Att:   {model.params['Q(\"DRY/A\")']:>+7.2f}")
print(f"Return TDs:           {model.params['Q(\"RET TD\")']:>+7.2f}")
print(f"Turnovers:            {model.params['TO']:>+7.2f}")
print(f"Defensive TOs:        {model.params['DTO']:>+7.2f}")
print(f"Penalty Differential: {model.params['PENDIF']:>+7.2f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Restore stdout and close file
sys.stdout = original_stdout
f.close()

print(f"\nOutput saved to: {output_file}")

# Create visualizations
print("\nGenerating Seahawks visualizations...")
plt.style.use('default')
seahawks_color = '#002244'  # Seahawks navy blue
seahawks_green = '#69BE28'  # Seahawks action green

# 1. Year-by-year actual vs predicted
fig1, ax1 = plt.subplots(figsize=(10, 6))
years = seahawks['Year'].values
x = np.arange(len(years))
width = 0.35

ax1.bar(x - width/2, seahawks['Margin'].values, width, label='Actual Margin', 
        color=seahawks_color, edgecolor='black', linewidth=1.5)
ax1.bar(x + width/2, seahawks['Predicted_Margin'].values, width, label='Predicted Margin',
        color=seahawks_green, edgecolor='black', linewidth=1.5, alpha=0.8)

ax1.set_xlabel('Season', fontsize=12, fontweight='bold')
ax1.set_ylabel('Point Differential', fontsize=12, fontweight='bold')
ax1.set_title('Seahawks: Actual vs Predicted Point Margin (2014-2017)', fontsize=14, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(years.astype(int))
ax1.legend(loc='upper right', fontsize=11)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax1.grid(axis='y', alpha=0.3)
plt.tight_layout()
margin_plot = os.path.join(outputs_dir, "seahawks_margin.png")
plt.savefig(margin_plot, dpi=300, bbox_inches='tight')
print(f"  ✓ Margin comparison: {margin_plot}")
plt.close()

# 2. Key stats over time
fig2, ((ax2a, ax2b), (ax2c, ax2d)) = plt.subplots(2, 2, figsize=(14, 10))

# Passing efficiency
ax2a.plot(years, seahawks['PY/A'].values, marker='o', linewidth=2.5, markersize=8, 
          color=seahawks_color, label='Offense')
ax2a.plot(years, seahawks['DPY/A'].values, marker='s', linewidth=2.5, markersize=8,
          color=seahawks_green, label='Defense (allowed)')
ax2a.set_title('Passing Yards per Attempt', fontweight='bold', fontsize=12)
ax2a.set_ylabel('Yards/Attempt', fontsize=10)
ax2a.legend()
ax2a.grid(alpha=0.3)

# Rushing efficiency
ax2b.plot(years, seahawks['RY/A'].values, marker='o', linewidth=2.5, markersize=8,
          color=seahawks_color, label='Offense')
ax2b.plot(years, seahawks['DRY/A'].values, marker='s', linewidth=2.5, markersize=8,
          color=seahawks_green, label='Defense (allowed)')
ax2b.set_title('Rushing Yards per Attempt', fontweight='bold', fontsize=12)
ax2b.set_ylabel('Yards/Attempt', fontsize=10)
ax2b.legend()
ax2b.grid(alpha=0.3)

# Turnovers
ax2c.plot(years, seahawks['TO'].values, marker='o', linewidth=2.5, markersize=8,
          color='#C8102E', label='Turnovers (bad)')
ax2c.plot(years, seahawks['DTO'].values, marker='s', linewidth=2.5, markersize=8,
          color='#00B140', label='Forced TOs (good)')
ax2c.set_title('Turnovers', fontweight='bold', fontsize=12)
ax2c.set_ylabel('Count', fontsize=10)
ax2c.set_xlabel('Season', fontsize=10)
ax2c.legend()
ax2c.grid(alpha=0.3)

# Special teams & penalties
ax2d.bar(years, seahawks['RET TD'].values, width=0.4, label='Return TDs',
         color=seahawks_color, edgecolor='black', linewidth=1)
ax2_twin = ax2d.twinx()
ax2_twin.plot(years, seahawks['PENDIF'].values, marker='D', linewidth=2.5, markersize=8,
              color='orange', label='Penalty Diff')
ax2d.set_title('Special Teams & Penalties', fontweight='bold', fontsize=12)
ax2d.set_ylabel('Return TDs', fontsize=10, color=seahawks_color)
ax2d.set_xlabel('Season', fontsize=10)
ax2_twin.set_ylabel('Penalty Differential', fontsize=10, color='orange')
ax2_twin.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax2d.legend(loc='upper left')
ax2_twin.legend(loc='upper right')
ax2d.grid(alpha=0.3)

fig2.suptitle('Seahawks Performance Metrics Over Time', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
stats_plot = os.path.join(outputs_dir, "seahawks_stats_timeline.png")
plt.savefig(stats_plot, dpi=300, bbox_inches='tight')
print(f"  ✓ Stats timeline: {stats_plot}")
plt.close()

# 3. Residuals plot
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.bar(years, seahawks['Residual'].values, color=[seahawks_green if r > 0 else '#C8102E' 
        for r in seahawks['Residual'].values], edgecolor='black', linewidth=1.5)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
ax3.set_xlabel('Season', fontsize=12, fontweight='bold')
ax3.set_ylabel('Residual (Actual - Predicted)', fontsize=12, fontweight='bold')
ax3.set_title('Seahawks: Over/Under Performance vs Model Prediction', fontsize=14, fontweight='bold', pad=20)
ax3.grid(axis='y', alpha=0.3)
# Add value labels
for i, (year, residual) in enumerate(zip(years, seahawks['Residual'].values)):
    ax3.text(year, residual, f'{residual:+.1f}', ha='center', 
             va='bottom' if residual > 0 else 'top', fontweight='bold', fontsize=10)
plt.tight_layout()
residual_plot = os.path.join(outputs_dir, "seahawks_residuals.png")
plt.savefig(residual_plot, dpi=300, bbox_inches='tight')
print(f"  ✓ Residuals plot: {residual_plot}")
plt.close()

print(f"\n✓ All Seahawks visualizations saved!")
print(f"\nAll outputs saved to: {outputs_dir}/")
print(f"\nTo view the images:")
print(f"  - Open the {outputs_dir}/ folder in VS Code")
print(f"  - Or use: eog {outputs_dir}/seahawks_*.png (Linux)")
print(f"  - Or open the {outputs_dir}/ folder in your file manager")
