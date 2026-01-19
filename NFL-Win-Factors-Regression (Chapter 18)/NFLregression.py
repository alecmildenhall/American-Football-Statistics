###########################################################################
### Chapter 18 - What Makes NFL Teams Win?                              ###
# Mathletics: How Gamblers, Managers, and Sports Enthusiasts              #
# Use Mathematics in Baseball, Basketball, and Football                   #
###########################################################################

import pandas as pd
import numpy as np
from statsmodels.formula.api import ols

nfldata = pd.read_csv("NFLdata.csv")

# full regression model
nflmodel = ols('Margin~Q("RET TD")+PENDIF+Q("PY/A")+Q("DPY/A")+Q("RY/A")+Q("DRY/A")+TO+DTO-1',data = nfldata).fit()
print(nflmodel.summary())

# passing-only regression
nflmodelpassing = ols('Margin~Q("PY/A")+Q("DPY/A")',data = nfldata).fit()
print(nflmodelpassing.summary())

# rushing-only regression
nflmodelrushing = ols('Margin~Q("RY/A")+Q("DRY/A")',data=nfldata).fit()
print(nflmodelrushing.summary())

# correlation matrix
print("++++++++++++++++++++++ Correlation Matrix ++++++++++++++++++++++")
corr = nfldata.drop(columns = ['Year','Team','Margin']).corr()

print(corr)
