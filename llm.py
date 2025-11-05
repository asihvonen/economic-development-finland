from openai import OpenAI
import pandas as pd
import os
from pathlib import Path
import tiktoken
from typing import Optional, Dict

# https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety
api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

MODEL="gpt-4o-mini"
PROMPT_TEMPLATE = """
You are an economic impact analysis assistant that specializes in the economy of Finland and its regions.
Given a scenario, identify:
1. The most affected region (x_1)
2. The most affected economic factor/industry (y_i)
3. The percentage change in that factor (x_2)
4. A qualitative description of how this affects GDP per capita and household disposable income in x_1.

You may use economic logic, regional specialization, and trade structure reasoning. 

You will be provided economic data which you will use for realism and to align your classifications accordingly. The economic data is structured as follows:
Region | Year | Economic factor 1 | Economic factor 2 | ... | Economic factor n
---|---|---|---|---|---
01 | 2015 | value_1_1 | value_1_2 | ... | value_1_n
...| ... | ... | ... | ... | ...
21 | 2022 | value_m_1 | value_m_2 | ... | value_m_n

Regions are labeled by their region codes (from 01 to 21) according to the Regions 2025 classification by Statistics Finland.

---

### Scenario
{scenario}

### Region specific?
{region_specific}

### (Optional) Economic Data Context
{data_context}

---

### Output Format (must be strictly JSON)
{{
    "most_affected_region": "string",
    "most_affected_factor": "string",
    "change_in_factor_percent": "float",
    "impact_summary": "string",
}}
"""

def analyze_scenario(
    scenario: str,
    region_specific: bool,
    data: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Run the LLM to analyze an economic scenario and extract structured output.
    Optionally uses data (industries, regions, etc.) as context.
    """
    data_context = None
    if data is not None:
        data_context = data.describe(include='all').to_string()

    prompt = PROMPT_TEMPLATE.format(
        scenario=scenario,
        region_specific="Yes" if region_specific else "No",
        data_context=data_context or "No dataset provided."
    )

    enc = tiktoken.encoding_for_model(MODEL)
    print(f"Length of prompt tokens:{len(enc.encode(prompt))}")

    if len(enc.encode(prompt)) > 10000:
        raise ValueError("Prompt too long for model context window.")
    response = client.responses.create(
        model=MODEL,
        input=prompt,
    )

    output = response.output[0].content[0].text
    return output

if __name__ == "__main__":
    # Example DataFrame with regional economic structure

    # parent path
    # data_path = Path().resolve() / "data" / "regional_economic_data.csv"
    # df = pd.read_csv(data_path)

    # test df
    df = pd.DataFrame({
        "Region": ["01", "02", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14"],
        "Year": [2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020],
        "Gross value added (millions of euro), A Agriculture, forestry and fishing": [
            270.3, 488, 305.2, 204.7, 437.4, 230.7, 149.6, 189.8, 476.7, 557.3, 378.9, 412.8, 473.4
        ],
    })

    scenario = "What happens if a new EU regulation limits the number of trees that can be cut down in Finland by 0.5%?"

    result = analyze_scenario(scenario, region_specific=False, data=df)

    print(result)