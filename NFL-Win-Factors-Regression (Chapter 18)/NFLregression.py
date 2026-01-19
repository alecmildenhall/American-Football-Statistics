###########################################################################
### Chapter 18 - What Makes NFL Teams Win?                              ###
# Mathletics: How Gamblers, Managers, and Sports Enthusiasts              #
# Use Mathematics in Baseball, Basketball, and Football                   #
###########################################################################

import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import os

# Create outputs folder if it doesn't exist
outputs_dir = "outputs"
os.makedirs(outputs_dir, exist_ok=True)

# Create output file (overwrites previous)
output_file = os.path.join(outputs_dir, "output.txt")

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

# Load NFL team performance data
# Variables:
#   RET TD  = Return touchdowns (kickoff/punt returns for TD)
#   PENDIF  = Penalty differential (team penalties - opponent penalties)
#   PY/A    = Passing yards per attempt (offensive)
#   DPY/A   = Defensive passing yards per attempt allowed
#   RY/A    = Rushing yards per attempt (offensive)
#   DRY/A   = Defensive rushing yards per attempt allowed
#   TO      = Turnovers committed by team
#   DTO     = Defensive turnovers (turnovers forced by defense)
#   Margin  = Point differential (points scored - points allowed)
nfldata = pd.read_csv("NFLdata.csv")

print("="*80)
print("NFL WIN FACTORS REGRESSION ANALYSIS")
print("Chapter 18: What Makes NFL Teams Win?")
print("="*80)
print()

# Full regression model
# Predicts point margin using all available factors:
# special teams, penalties, passing offense/defense, rushing offense/defense, and turnovers
# The "-1" removes the intercept (forcing the model through the origin)
print("\n" + "="*80)
print("1. FULL REGRESSION MODEL")
print("Including all factors: returns, penalties, passing, rushing, turnovers")
print("="*80)
nflmodel = ols('Margin~Q("RET TD")+PENDIF+Q("PY/A")+Q("DPY/A")+Q("RY/A")+Q("DRY/A")+TO+DTO-1',data = nfldata).fit()
print(nflmodel.summary())

# Passing-only regression
# Examines how much of point margin can be explained by passing efficiency alone
# Tests the hypothesis that "passing is more important than rushing" in modern NFL
print("\n" + "="*80)
print("2. PASSING-ONLY REGRESSION")
print("How much can passing offense/defense alone explain?")
print("="*80)
nflmodelpassing = ols('Margin~Q("PY/A")+Q("DPY/A")',data = nfldata).fit()
print(nflmodelpassing.summary())

# Rushing-only regression
# Examines how much of point margin can be explained by rushing efficiency alone
# Compare R² with passing model to see which dimension is more predictive
print("\n" + "="*80)
print("3. RUSHING-ONLY REGRESSION")
print("How much can rushing offense/defense alone explain?")
print("="*80)
nflmodelrushing = ols('Margin~Q("RY/A")+Q("DRY/A")',data=nfldata).fit()
print(nflmodelrushing.summary())

# Correlation matrix
# Shows pairwise correlations between all predictor variables
# Useful for identifying multicollinearity (high correlation between predictors)
# Values close to 1 or -1 indicate strong linear relationships
print("\n" + "="*80)
print("4. CORRELATION MATRIX")
print("Pairwise correlations between all predictor variables")
print("="*80)
corr = nfldata.drop(columns = ['Year','Team','Margin']).corr()

print(corr)
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Restore stdout and close file
sys.stdout = original_stdout
f.close()

print(f"\nOutput saved to: {output_file}")

# Create visualizations
print("\nGenerating visualizations...")
plt.style.use('default')

# 1. Correlation heatmap
fig1, ax1 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax1)
ax1.set_title('NFL Win Factors - Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
corr_plot = os.path.join(outputs_dir, "correlation_heatmap.png")
plt.savefig(corr_plot, dpi=300, bbox_inches='tight')
print(f"  ✓ Correlation heatmap: {corr_plot}")
plt.close()

# 2. Model coefficients comparison
fig2, ax2 = plt.subplots(figsize=(12, 6))
coefs = pd.DataFrame({
    'Full Model': nflmodel.params,
    'Passing Only': [nflmodelpassing.params.get(p, 0) for p in nflmodel.params.index],
    'Rushing Only': [nflmodelrushing.params.get(p, 0) for p in nflmodel.params.index]
})
coefs.plot(kind='bar', ax=ax2, color=['#2E86AB', '#A23B72', '#F18F01'])
ax2.set_title('Model Coefficients Comparison', fontsize=14, fontweight='bold')
ax2.set_xlabel('Predictors', fontsize=12)
ax2.set_ylabel('Coefficient Value (Points per Unit)', fontsize=12)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.legend(title='Model Type', loc='best')
ax2.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
coef_plot = os.path.join(outputs_dir, "coefficient_comparison.png")
plt.savefig(coef_plot, dpi=300, bbox_inches='tight')
print(f"  ✓ Coefficient comparison: {coef_plot}")
plt.close()

# 3. R-squared comparison
fig3, ax3 = plt.subplots(figsize=(8, 6))
r_squared = {
    'Full Model': nflmodel.rsquared,
    'Passing Only': nflmodelpassing.rsquared,
    'Rushing Only': nflmodelrushing.rsquared
}
colors = ['#2E86AB', '#A23B72', '#F18F01']
bars = ax3.bar(r_squared.keys(), r_squared.values(), color=colors, edgecolor='black', linewidth=1.5)
ax3.set_title('Model Performance Comparison (R²)', fontsize=14, fontweight='bold')
ax3.set_ylabel('R² (Proportion of Variance Explained)', fontsize=12)
ax3.set_ylim(0, 1)
ax3.grid(axis='y', alpha=0.3)
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)
plt.tight_layout()
r2_plot = os.path.join(outputs_dir, "r_squared_comparison.png")
plt.savefig(r2_plot, dpi=300, bbox_inches='tight')
print(f"  ✓ R² comparison: {r2_plot}")
plt.close()

# 4. Actual vs Predicted (Full Model)
fig4, ax4 = plt.subplots(figsize=(10, 8))
predicted = nflmodel.predict(nfldata)
ax4.scatter(nfldata['Margin'], predicted, alpha=0.6, s=50, color='#2E86AB', edgecolor='black', linewidth=0.5)
ax4.plot([nfldata['Margin'].min(), nfldata['Margin'].max()], 
         [nfldata['Margin'].min(), nfldata['Margin'].max()], 
         'r--', linewidth=2, label='Perfect Prediction')
ax4.set_xlabel('Actual Margin (Points)', fontsize=12)
ax4.set_ylabel('Predicted Margin (Points)', fontsize=12)
ax4.set_title('Full Model: Actual vs Predicted Point Margins', fontsize=14, fontweight='bold')
ax4.legend(loc='upper left', fontsize=10)
ax4.grid(alpha=0.3)
plt.tight_layout()
actual_pred_plot = os.path.join(outputs_dir, "actual_vs_predicted.png")
plt.savefig(actual_pred_plot, dpi=300, bbox_inches='tight')
print(f"  ✓ Actual vs Predicted: {actual_pred_plot}")
plt.close()

print(f"\n✓ All visualizations saved!")
print(f"\nAll outputs saved to: {outputs_dir}/")
print(f"\nTo view the images:")
print(f"  - Open the {outputs_dir}/ folder in VS Code")
print(f"  - Or use: eog {outputs_dir}/*.png (Linux)")
print(f"  - Or open the {outputs_dir}/ folder in your file manager")
