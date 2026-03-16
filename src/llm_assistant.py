import os
from openai import OpenAI


def explain_transaction(transaction_data: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return (
            "OpenAI API key is not configured. "
            "Set OPENAI_API_KEY in your environment to enable the Fraud Assistant."
        )

    try:
        client = OpenAI(api_key=api_key)

        prompt = f"""
You are a banking fraud analyst.

Analyze the following transaction and explain whether it looks suspicious.

Transaction data:
{transaction_data}

Explain:
1. possible fraud indicators
2. risk level
3. recommended next action
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial fraud expert."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Fraud Assistant is unavailable right now: {e}"