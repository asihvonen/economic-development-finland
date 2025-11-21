from curses import raw
from openai import OpenAI
import pandas as pd
import os
from pathlib import Path
import tiktoken
from typing import Optional, Dict

# https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety

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

class LLM:
    def __init__(self):
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o-mini"
        self.economic_data = pd.read_csv(Path.cwd() / "data" / "regional_economic_data.csv")

    def analyze_scenario(
        self,
        scenario: str,
        region_specific: bool= False,
    ) -> Dict:
        """
        Returns a normalized dict with:
        - most_affect_factor (str)
        - change_in_factor_percent(str)
        - impact_summary (str)
        """
        data_context = self.economic_data.describe(include='all').to_string()

        prompt = PROMPT_TEMPLATE.format(
            scenario=scenario,
            region_specific="Yes" if region_specific else "No",
            data_context=data_context,
        )

        enc = tiktoken.encoding_for_model(self.model)
        print(f"Length of prompt tokens:{len(enc.encode(prompt))}")

        if len(enc.encode(prompt)) > 10000:
            raise ValueError("Prompt too long for model context window.")
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        raw_output = response.to_dict()["content"]

        most = None
        change = None
        summary = None

        # Normalize different possible raw formats
        if isinstance(raw_output, dict):
            most = raw_output.get("most_affected_factor") or raw_output.get("factor") or raw_output.get("industry") or raw_output.get("most_affected")
            change = raw_output.get("change_in_factor_percent") or raw_output.get("change") or raw_output.get("percent_change")
            summary = raw_output.get("impact_summary") or raw_output.get("summary") or raw_output.get("explanation")
        elif isinstance(raw_output, (list, tuple)):
            if len(raw_output) >= 1:
                most = raw_output[0]
            if len(raw_output) >= 2:
                change = raw_output[1]
            if len(raw_output) >= 3:
                summary = raw_output[2]
        elif isinstance(raw_output, (int, float)):
            # numeric -> treat as percent change, no factor known
            change = f"{raw_output}%"
        elif isinstance(raw_output, str):
            # If the model returned a freeform string, use it as the summary
            summary = raw

        # Ensure change is a percent string like '5%' or '-2.5%'
        if isinstance(change, (int, float)):
            change = f"{change}%"
        elif isinstance(change, str):
            s = change.strip()
            # if it's a bare number, append '%'
            try:
                # allow existing '%' but also accept plain numbers
                _ = float(s.replace("%", ""))
                if not s.endswith("%"):
                    s = s + "%"
                change = s
            except Exception:
                change = s  # leave as-is if not numeric-like

        most = most or "unknown"
        change = change or "0%"
        summary = summary or f"Predicted {change} change in {most}."

        return {
            "most_affected_factor": most,
            "change_in_factor_percent": change,
            "impact_summary": summary
        }