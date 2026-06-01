# analysis/07_personalized_prompts.py
"""Generate personalized code based on developer cognitive state."""
import json
from pathlib import Path

class PersonalizedCodingAssistant:
    """
    AI assistant that adapts to developer's cognitive state.
    This extends Kovalchuk et al.'s work into practical application.
    """
    
    def __init__(self, cognitive_profile_path):
        """Load cognitive profiles for personalization."""
        import pandas as pd
        self.profiles = pd.read_csv(cognitive_profile_path)
    
    def get_personalization_config(self, developer_name):
        """Get personalization settings based on developer's cognitive profile."""
        
        profile = self.profiles[self.profiles['evaluator_name'] == developer_name]
        
        if len(profile) == 0:
            return self._default_config()
        
        expertise = profile['expertise_level'].iloc[0]
        understanding = profile['understanding_mean'].iloc[0]
        intention_rate = profile['intention_rate'].iloc[0]
        
        config = {
            'expertise_level': expertise,
            'explanation_detail': self._get_explanation_level(expertise, understanding),
            'code_comment_density': self._get_comment_density(expertise),
            'suggestion_confidence': self._get_confidence_threshold(intention_rate),
            'preferred_model': 'gemini' if profile['gemini_preference'].iloc[0] else 'groq'
        }
        
        return config
    
    def _get_explanation_level(self, expertise, understanding):
        """Determine how much explanation to provide."""
        if expertise in ['student_bachelor', 'junior']:
            return 'high'  # Need more explanation
        elif understanding < 4.0:
            return 'medium'  # Some confusion detected
        else:
            return 'low'  # Expert, minimal explanation
    
    def _get_comment_density(self, expertise):
        """Determine code comment density."""
        if expertise == 'student_bachelor':
            return 'detailed'  # Line-by-line comments
        elif expertise == 'junior':
            return 'moderate'  # Block comments
        else:
            return 'minimal'  # Only complex sections
    
    def _get_confidence_threshold(self, intention_rate):
        """Determine confidence needed for suggestions."""
        if intention_rate < 0.5:
            return 'high'  # Need very confident suggestions
        else:
            return 'normal'
    
    def _default_config(self):
        """Default configuration."""
        return {
            'expertise_level': 'unknown',
            'explanation_detail': 'medium',
            'code_comment_density': 'moderate',
            'suggestion_confidence': 'normal',
            'preferred_model': 'gemini'
        }
    
    def personalize_prompt(self, original_prompt, developer_name):
        """Modify prompt based on developer's cognitive profile."""
        
        config = self.get_personalization_config(developer_name)
        
        personalized = f"""
[Personalized for {config['expertise_level']} developer]

{original_prompt}

Style Requirements:
- Explanation detail: {config['explanation_detail']}
- Code comments: {config['code_comment_density']}

Generate code accordingly.
"""
        return personalized
    
    def generate_personalized_code(self, original_prompt, developer_name, llm_function):
        """Generate code personalized to developer's cognitive state."""
        
        personalized_prompt = self.personalize_prompt(original_prompt, developer_name)
        
        # Use the developer's preferred model
        model_to_use = config['preferred_model']
        
        # Call LLM with personalized prompt
        generated_code = llm_function(personalized_prompt)
        
        return {
            'generated_code': generated_code,
            'used_model': model_to_use,
            'personalization_config': config
        }

# Example usage
if __name__ == "__main__":
    assistant = PersonalizedCodingAssistant('results/cognitive_profiles.csv')
    
    # Test with different developer profiles
    for developer in ['Joullanar Ali', 'Egor']:
        config = assistant.get_personalization_config(developer)
        print(f"\n{developer}:")
        print(f"  Explanation: {config['explanation_detail']}")
        print(f"  Comments: {config['code_comment_density']}")
        print(f"  Preferred model: {config['preferred_model']}")