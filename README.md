# Economic Development Modeling for Finland

A scenario analysis tool designed to analyse the Economic Development of Finland

## Overview

This project provides a comprehensive tool for visualising and predicting economic trends across Finnish regions. Built in collaboration with OP Financial Group, it enables users to explore regional differences, compare economic indicators, and simulate the impact of various economic scenarios.

### Key Features

- **Interactive Regional Maps**: Visual representations of key economic indicators across 19 Finnish regions
- **Scenario Simulation**: Test "what-if" scenarios to see how different economic events might affect regions
- **Custom Economic Indices**: Specialized measures for Labour & Productivity and Economic Structure & Human Capital
- **Regional Dashboards**: Detailed economic profiles for each region, including dominant industries and development trends
- **Models**: Multiple machine learning models designed to fit the data
## What This Tool Does

The application helps users understand:
- How different Finnish regions are developing economically
- Which regions are most vulnerable to specific economic shocks
- How events like energy price increases or trade disruptions might affect different areas
- Regional differences in economic structure and workforce composition

## Project Structure
```
├── data/                                 # Economic datasets and statistics
├── visualizations/                       # Interactive map HTML files
├── src/                                # Web interface
```

## Data Sources

All data comes from publicly available Finnish sources:

- **Statistics Finland (stat.fi)**
- **Tulli (tulli.fi)**

The dataset covers 19 Finnish regions from 2000-2022 with 85 different economic indicators.

## How It Works

### Predictions

The tool uses **Ridge Linear Regression** to predict GDP per capita and household disposable income. This approach was chosen because:
- It works well with limited data
- It's simple and easy to interpret
- It handles relationships between different economic factors effectively

Each region gets its own custom model that captures local economic patterns. The models achieve very high accuracy (typically 95-99% R²).

### Economic Indices

Two composite indices help compare regions:

**Labour & Productivity Index** measures:
- Employment and unemployment rates
- Job availability
- Economic efficiency

**Economic Structure & Human Capital Index** captures:
- Education levels
- Knowledge-intensive employment
- Industry diversification

### Scenario Analysis

When you input a scenario (like "What if tourism grows by 20%?"):

1. An AI analyzes your scenario to identify the most affected industry
2. The system calculates how this change spreads to related industries
3. Regional impacts are calculated based on each region's economic structure
4. The map updates to show predicted changes across Finland

## Getting Started

### Installation
```bash
pip install flask pandas numpy plotly geopandas folium branca openai tiktoken
```

### Setup

Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

### Running the App
```bash
python app.py
```

Open your browser to `http://localhost:5000`

### Using the Interface

1. **Explore Maps**: Choose different economic indicators from the dropdown
2. **View Trends**: Use the time slider to see how regions changed over the years
3. **Test Scenarios**: 
   - Switch to "Map + Chat" mode
   - Describe an economic scenario
   - See how it would affect different regions

## Example Scenarios

The tool includes pre-defined scenarios for common economic events:

- **Declining wood product demand**: How would a drop in forestry exports affect Finland?
- **High energy costs**: Which regions suffer most from expensive electricity?
- **Rising interest rates**: How do housing markets and construction respond?
- **Labour shortages**: Impact of an aging workforce on different regions
- **Trade tariffs**: Effects of reduced exports to major markets

## Important Notes

- This is a research/educational tool, not for real business decisions
- The AI-generated insights should be verified before acting on them
- Limited historical data means predictions have uncertainty
- All data is publicly available and anonymized

## Future Enhancements

- Real-time data updates via APIs
- Longer-term scenario effects
- Municipality-level detail
- Validation against actual economic outcomes

## Documentation

For detailed technical information, methodology, and results, please refer to the project presentation and final report PDFs included in this repository.

## Acknowledgments

This project was completed as part of Aalto University's Data Science Project course (CS-C3250) in collaboration with OP Financial Group.
