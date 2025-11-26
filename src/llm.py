from curses import raw
import json
import re
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
    1. The most affected industry in the country
    2. The percentage change in that industry due to the scenario
    3. A qualitative description of how this will generally affect GDP per capita and household disposable income in the regions of Finland.

    You will use economic logic, regional specialization, and the economic data provided as context to inform your analysis. 

    ---

    ### Scenario
    {scenario}

    ### Economic Data Context
    {data_context}

    ---

    ### Output Format (JSON)
    The output should be a JSON object formatted as follows:
    {{
    "most_affected_industry": "string",
    "change_in_industry_percent": "string",
    "impact_summary": "string",
    }}

    most_affected_industry: The industry most impacted by the scenario. The industry must match exactly one of the industry names from the economic data context.
    change_in_industry_percent: The percentage change in that industry as a string with a '%' sign.
    impact_summary: A paragraph on the expected impact on GDP per capita and household disposable income across Finland and its regions.

"""

class LLM:
    def __init__(self):
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-5-mini-2025-08-07"  #"gpt-4o-mini"
        self.context_window = 400000  # 128000 - tokens, determined by model
        self.economic_data = pd.read_csv(Path.cwd().parent / "economic-development-finland" / "data" / "economic_data_context.csv").to_markdown()

    def analyze_scenario(
        self,
        scenario: str,
    ) -> tuple[str, str]:
        """
        Returns a normalized dict with:
        - most_affected_industry (str)
        - change_in_industry_percent(str)
        - impact_summary (str)
        """

        prompt = PROMPT_TEMPLATE.format(
            scenario=scenario,
            data_context=self.economic_data,
        )

        enc = tiktoken.encoding_for_model(self.model)
        if len(enc.encode(prompt)) > self.context_window:
            raise ValueError(f"Prompt of length {len(enc.encode(prompt))} tokens is too long for model context window of {self.context_window} tokens.")
        
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        # get text output from response
        response_dict = response.to_dict()
        output_text = None
        try:
            outputs = response_dict.get("output", [])
            for item in outputs:
                for content in item.get("content", []):
                    if content.get("type") == "output_text" and content.get("text"):
                        output_text = content.get("text")
                        break
                if output_text:
                    break
        except Exception as error:
            raise ValueError(f"Could not extract output text from LLM response: {error}")

        # extract JSON object from output_text
        json_text = None
        match = re.search(r"```(?:json)?\s*({.*?})\s*```", output_text, re.DOTALL)
        if match:
            json_text = match.group(1)
        else:
            # try to find first { ... } in text
            match = re.search(r"({.*})", output_text, re.DOTALL)
            if match:
                json_text = match.group(1)

        if not json_text:
            raise ValueError(f"Could not find JSON object from LLM output. Output: {output_text}")
        
        try:
            parsed = json.loads(json_text)
        except Exception as error:
            raise ValueError(f"Could not parse JSON from output text: {error}.")
        
        industry = parsed.get("most_affected_industry")
        change = parsed.get("change_in_industry_percent")
        summary = parsed.get("impact_summary")

        return (industry, change, summary)