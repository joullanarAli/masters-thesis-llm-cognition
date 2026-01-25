# Functions to call AI APIs
from google import genai
import json
import time
from typing import Dict, Optional
import requests


# def generate_with_google(prompt: str, config: dict) -> str:
#     """
#     Generate code using the official Google Gemini API.
#     Uses the model and key from the config dictionary.
#     """
#     try:
#         # 1. Initialize client with the key from your YAML file
#         client = genai.Client(api_key=config['api_key'])
        
#         # 2. Get the model from config (defaults to gemini-3-flash-preview)
#         model_id = config.get('model', 'gemini-3-flash-preview')
        
#         # 3. Make the API call
#         response = client.models.generate_content(
#             model=model_id,
#             contents=prompt,
#             config={
#                 "temperature": 0.7,  # Controls randomness. Keep low for consistent code.
#                 "max_output_tokens": config.get('max_tokens', 5000),
#             }
#         )
#         return response.text
        
#     except Exception as e:
#         # If the API call fails, log it and return an error placeholder
#         print(f"[ERROR - Google Gemini] {e}")
#         return f"# API Error: {str(e)[:100]}"

def generate_with_google(prompt: str, config: dict) -> str:
    """
    Generate code using the official Google Gemini API with improved error handling.
    """
    from google import genai
    import time
    import re

    client = genai.Client(api_key=config['api_key'])
    model_id = config.get('model', 'gemini-3-flash-preview')

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config={
                    "temperature": 0.7,
                    "max_output_tokens": 2000,
                }
            )
            return response.text

        except Exception as e:
            error_msg = str(e)
            print(f"[ATTEMPT {attempt + 1}/{max_retries}] API Error: {error_msg[:150]}")

            # CRITICAL: Stop retrying on 403 Forbidden errors (permission issue)
            if "403 Forbidden" in error_msg or "PERMISSION_DENIED" in error_msg:
                print("   ✗ Fatal 403/Permission error. Stopping retries. Check API key and project setup.")
                return f"# API Error: 403 Forbidden. Check project & billing setup."

            # Handle 429 Rate Limit with extracted retry time
            if "RESOURCE_EXHAUSTED" in error_msg:
                delay_match = re.search(r'retryDelay[\'"]?\s*:\s*[\'"]?(\d+)', error_msg)
                wait_time = int(delay_match.group(1)) if delay_match else 30
                print(f"   Rate limit hit. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                # For other transient errors
                wait_time = (2 ** attempt) * 10  # 10, 20, 40 seconds
                print(f"   Other error. Waiting {wait_time} seconds...")
                time.sleep(wait_time)

    # If all retries fail (excluding 403)
    return f"# API Error: Request failed after {max_retries} attempts."