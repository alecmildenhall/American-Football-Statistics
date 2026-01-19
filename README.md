# American Football Statistics

Statistical analysis projects from *Mathletics: How Gamblers, Managers, and Sports Enthusiasts Use Mathematics in Baseball, Basketball, and Football*.

## Project Structure

- **NFL-Win-Factors-Regression (Chapter 18)**: Regression analysis examining what factors contribute to NFL team wins
- **Chapter-19**: QB rating analysis
- **Chapter-21**: Field goals, pass/run decisions, and College World Series analysis
- **Chapter-23**: Dynamic programming charts and NFL PAT analysis
- **Chapter-24**: Additional PAT analysis
- **Chapter-25**: NFL draft and overtime analysis
- **Chapter-26**: Draft data
- **Chapter-27**: Convex hull analysis and passing location detection

## Setup

1. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Scripts

Each chapter folder contains Python and/or R scripts for analysis. To run a Python script:

```bash
cd <chapter-folder>
python <script-name>.py
```

For example:
```bash
cd NFL-Win-Factors-Regression\ (Chapter\ 18)
python NFLregression.py
```

## Requirements

- Python 3.12+
- pandas
- numpy
- statsmodels
- (Additional packages listed in requirements.txt)

## Data

Each chapter folder contains its own CSV or data files needed for the analysis.
