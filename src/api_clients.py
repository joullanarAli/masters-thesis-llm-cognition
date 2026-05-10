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
                    "max_output_tokens": 8192,
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

# Add to api_client.py (after your Google function)
# api_client.py - Add this working Qwen function

def generate_with_qwen(prompt: str, config: dict) -> str:
    """
    Generate code using a smaller, freely available Qwen model.
    Uses 'Qwen/Qwen2-1.5B-Instruct' which is open and works without authentication.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        # Cache model globally
        if not hasattr(generate_with_qwen, "model"):
            print("   Loading Qwen2-1.5B model (small, ~3GB RAM, first time takes ~1 minute)...")
            # Using a smaller, open model that doesn't require authentication
            model_name = config.get('model', 'Qwen/Qwen2-1.5B-Instruct')
            
            generate_with_qwen.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            generate_with_qwen.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            print("   ✓ Qwen model loaded successfully!")
        
        messages = [
            {"role": "system", "content": "You are a Python code generator. Complete the function implementation. Return ONLY the code, no explanations."},
            {"role": "user", "content": prompt}
        ]
        
        text = generate_with_qwen.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = generate_with_qwen.tokenizer([text], return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        outputs = generate_with_qwen.model.generate(
            **inputs,
            max_new_tokens=config.get('max_tokens', 512),
            temperature=config.get('temperature', 0.2),
            do_sample=True,
            top_p=0.95
        )
        
        generated = generate_with_qwen.tokenizer.decode(outputs[0], skip_special_tokens=True)
        code = generated[len(text):].strip()
        
        # Clean up markdown
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        
        return code.strip()
        
    except ImportError as e:
        return f"# ERROR: Missing dependencies. Run: pip install transformers torch accelerate\n# {str(e)}"
    except Exception as e:
        print(f"   Qwen Error: {e}")
        return f"# API Error (Qwen): {str(e)[:100]}"
    
# Add this to your api_client.py file

def generate_with_codegen(prompt: str, config: dict) -> str:
    """
    Generate code using Salesforce CodeGen-350M.
    Small model (350M parameters), downloads quickly (~1.4GB).
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        # Cache model globally to avoid reloading
        if not hasattr(generate_with_codegen, "model"):
            print("   Downloading CodeGen-350M (first time, ~1.4GB, 2-5 minutes)...")
            model_name = config.get('model', 'Salesforce/codegen-350M-mono')
            
            generate_with_codegen.tokenizer = AutoTokenizer.from_pretrained(model_name)
            generate_with_codegen.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            print("   ✓ CodeGen model loaded successfully!")
        
        # Format prompt for code generation
        formatted_prompt = f"# Complete the following Python function:\n{prompt}\n\n# Solution:\n"
        
        inputs = generate_with_codegen.tokenizer(formatted_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        outputs = generate_with_codegen.model.generate(
            **inputs,
            max_new_tokens=config.get('max_tokens', 512),
            temperature=config.get('temperature', 0.2),
            do_sample=True,
            pad_token_id=generate_with_codegen.tokenizer.eos_token_id
        )
        
        response = generate_with_codegen.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract code after the solution marker
        if "# Solution:" in response:
            code = response.split("# Solution:")[-1].strip()
        else:
            code = response.strip()
        
        # Clean up any markdown
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        
        return code
        
    except ImportError as e:
        return f"# ERROR: Missing dependencies. Run: pip install transformers torch\n# {str(e)}"
    except Exception as e:
        print(f"   CodeGen Error: {e}")
        return f"# API Error (CodeGen): {str(e)[:100]}"
def generate_with_starcoder(prompt: str, config: dict) -> str:
    """
    Generate code using StarCoder-1B.
    1B parameters, good quality, reasonable download.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        if not hasattr(generate_with_starcoder, "model"):
            print("   Loading StarCoder-1B (~2GB, 5-10 min download)...")
            model_name = config.get('model', 'bigcode/starcoderbase-1b')
            
            generate_with_starcoder.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            generate_with_starcoder.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            print("   ✓ StarCoder loaded!")
        
        # Format prompt for code completion
        formatted_prompt = f"<fim_prefix>{prompt}<fim_suffix>\n<fim_middle>"
        
        inputs = generate_with_starcoder.tokenizer(formatted_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        outputs = generate_with_starcoder.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True
        )
        
        response = generate_with_starcoder.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the generated code
        if "<fim_middle>" in response:
            code = response.split("<fim_middle>")[-1].split("<fim_prefix>")[0]
        else:
            code = response
        
        return code.strip()
        
    except Exception as e:
        return f"# Error: {e}"


def generate_with_deepseek(prompt: str, config: dict) -> str:
    """
    Generate code using DeepSeek-Coder-6.7B via Hugging Face.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        if not hasattr(generate_with_deepseek, "model"):
            print("   Loading DeepSeek-Coder-6.7B model (first time may take 2-3 minutes)...")
            model_name = config.get('model', 'deepseek-ai/deepseek-coder-6.7b-instruct')
            
            generate_with_deepseek.tokenizer = AutoTokenizer.from_pretrained(model_name)
            generate_with_deepseek.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            print("   Model loaded successfully!")
        
        messages = [
            {"role": "user", "content": f"### Instruction:\nComplete the following Python function. Return ONLY the code, no explanation.\n\n{prompt}\n\n### Response:\n```python"}
        ]
        
        inputs = generate_with_deepseek.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(generate_with_deepseek.model.device)
        
        outputs = generate_with_deepseek.model.generate(
            inputs,
            max_new_tokens=config.get('max_tokens', 512),
            temperature=config.get('temperature', 0.2),
            do_sample=False
        )
        
        response = generate_with_deepseek.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract code from response
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0]
        elif "```" in response:
            code = response.split("```")[1].split("```")[0]
        else:
            code = response
        
        return code.strip()
        
    except ImportError as e:
        return f"# ERROR: Missing dependency. Run: pip install transformers torch\n# {str(e)}"
    except Exception as e:
        print(f"   DeepSeek API Error: {e}")
        return f"# API Error (DeepSeek): {str(e)[:100]}"


def generate_with_codellama(prompt: str, config: dict) -> str:
    """
    Generate code using CodeLlama-7B via Hugging Face.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        if not hasattr(generate_with_codellama, "model"):
            print("   Loading CodeLlama-7B model (first time may take 2-3 minutes)...")
            model_name = config.get('model', 'codellama/CodeLlama-7b-Python-hf')
            
            generate_with_codellama.tokenizer = AutoTokenizer.from_pretrained(model_name)
            generate_with_codellama.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("   Model loaded successfully!")
        
        # CodeLlama uses a different prompt format
        formatted_prompt = f"[INST] Write a complete Python function that solves the following problem. Return ONLY the code, no explanations.\n\n{prompt} [/INST]\n```python\n"
        
        inputs = generate_with_codellama.tokenizer(formatted_prompt, return_tensors="pt").to(generate_with_codellama.model.device)
        
        outputs = generate_with_codellama.model.generate(
            **inputs,
            max_new_tokens=config.get('max_tokens', 512),
            temperature=config.get('temperature', 0.2),
            do_sample=True
        )
        
        response = generate_with_codellama.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the generated part
        code = response.split("[/INST]")[-1].strip()
        
        # Clean up markdown
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        
        return code.strip()
        
    except ImportError as e:
        return f"# ERROR: Missing dependency. Run: pip install transformers torch\n# {str(e)}"
    except Exception as e:
        print(f"   CodeLlama API Error: {e}")
        return f"# API Error (CodeLlama): {str(e)[:100]}"
    
def generate_with_phi2(prompt: str, config: dict) -> str:
    """
    Generate code using Microsoft's Phi-2 (2.7B parameters).
    Smaller than DeepSeek, better than CodeGen.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        if not hasattr(generate_with_phi2, "model"):
            print("   Downloading Phi-2 (~5GB, 10-20 minutes)...")
            model_name = config.get('model', 'microsoft/phi-2')
            
            generate_with_phi2.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True,
                pad_token='<|endoftext|>'
            )
            generate_with_phi2.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            print("   ✓ Phi-2 loaded!")
        
        # Phi-2 prompt format
        formatted_prompt = f"Instruct: Write a complete Python function.\n{prompt}\nOutput:\n```python\n"
        
        inputs = generate_with_phi2.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=1024)
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        outputs = generate_with_phi2.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True,
            pad_token_id=generate_with_phi2.tokenizer.eos_token_id
        )
        
        response = generate_with_phi2.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract code
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0]
        else:
            code = response.split("Output:")[-1].strip()
        
        return code.strip()
        
    except Exception as e:
        return f"# Error: {e}"
    
# Add this to your api_client.py

def generate_with_groq(prompt: str, config: dict) -> str:
    """
    Generate code using Groq's free API (Mixtral 8x7B).
    Fast, free, and good quality for code generation.
    """
    try:
        from groq import Groq
        
        # Initialize client
        client = Groq(api_key=config['api_key'])
        
        # Format prompt for code generation
        system_prompt = "You are a Python code generator. Complete the function implementation. Return ONLY the code, no explanations or markdown formatting."
        
        response = client.chat.completions.create(
            model=config.get('model', 'mixtral-8x7b-32768'),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=config.get('temperature', 0.2),
            max_tokens=config.get('max_tokens', 500),
            top_p=0.95
        )
        
        code = response.choices[0].message.content.strip()
        
        # Clean up any markdown if present
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        
        return code.strip()
        
    except ImportError as e:
        return f"# ERROR: Missing groq package. Run: pip install groq\n# {str(e)}"
    except Exception as e:
        print(f"   Groq API Error: {e}")
        return f"# API Error (Groq): {str(e)[:100]}"