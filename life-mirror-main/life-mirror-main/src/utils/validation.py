from guardrails import Guard
from functools import wraps
from typing import Any, Dict, Optional
import json
from src.schemas.guardrails import (
    validate_content_safety, sanitize_output,
    create_fashion_guard, create_bio_guard, create_comparison_guard
)

def guardrails_validate(input_schema, output_schema):
    """Enhanced Guardrails validation with safety checks"""
    guard = Guard.from_pydantic(output_schema)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            try:
                # Validate output structure
                validated_output = guard.parse(result.dict())
                
                # Additional safety validation
                output_text = json.dumps(result.dict())
                is_safe, issues = validate_content_safety(output_text)
                
                if not is_safe:
                    # Apply content sanitization
                    sanitized_result = apply_content_sanitization(result.dict())
                    validated_output = guard.parse(sanitized_result)
                    validated_output.warnings = getattr(validated_output, 'warnings', []) + [
                        "Content was sanitized for safety"
                    ]
                
                return validated_output
                
            except Exception as e:
                # If validation fails, return original with warning
                result_dict = result.dict() if hasattr(result, 'dict') else result
                result_dict['validation_warning'] = f"Validation error: {str(e)}"
                return result_dict
                
        return wrapper
    return decorator

def apply_content_sanitization(data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply content sanitization to nested dictionary data"""
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = sanitize_output(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    sanitize_output(item) if isinstance(item, str) else item
                    for item in value
                ]
            elif isinstance(value, dict):
                sanitized[key] = apply_content_sanitization(value)
            else:
                sanitized[key] = value
        return sanitized
    return data

def validate_llm_output(output_text: str, agent_type: str) -> tuple[bool, Optional[str], Optional[Dict]]:
    """
    Validate LLM output for specific agent types.
    
    Returns:
        (is_valid, error_message, sanitized_output)
    """
    try:
        # Parse JSON output
        try:
            parsed_output = json.loads(output_text)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {str(e)}", None
        
        # Agent-specific validation
        if agent_type == "fashion":
            guard = create_fashion_guard()
        elif agent_type == "bio":
            guard = create_bio_guard()
        elif agent_type == "comparison":
            guard = create_comparison_guard()
        else:
            # Generic validation
            is_safe, issues = validate_content_safety(output_text)
            if not is_safe:
                sanitized = apply_content_sanitization(parsed_output)
                return True, None, sanitized
            return True, None, parsed_output
        
        # Validate with Guardrails
        validated_output = guard.parse(parsed_output)
        
        # Safety check
        is_safe, issues = validate_content_safety(output_text)
        if not is_safe:
            sanitized = apply_content_sanitization(parsed_output)
            return True, f"Content sanitized: {'; '.join(issues)}", sanitized
        
        return True, None, validated_output.dict() if hasattr(validated_output, 'dict') else validated_output
        
    except Exception as e:
        return False, f"Validation error: {str(e)}", None

def create_fallback_response(agent_type: str, error_message: str) -> Dict[str, Any]:
    """Create a safe fallback response for failed validations"""
    
    base_response = {
        "success": True,
        "validation_fallback": True,
        "original_error": error_message
    }
    
    if agent_type == "fashion":
        base_response["data"] = {
            "outfit_rating": 50,
            "items": [],
            "good": ["Shows personal style"],
            "bad": [],
            "improvements": ["Consider experimenting with different styles"],
            "confidence": 0.3
        }
    elif agent_type == "bio":
        base_response["data"] = {
            "vibe_summary": "Shows authentic personality",
            "strengths": ["Clear communication"],
            "weaknesses": [],
            "improvements": ["Consider adding more detail"],
            "confidence": 0.3
        }
    elif agent_type == "comparison":
        base_response["data"] = {
            "similarities": ["Shares common presentation elements"],
            "differences": ["Unique personal style"],
            "insights": ["Individual approach to presentation"],
            "recommendations": ["Continue developing personal style"],
            "confidence": 0.3
        }
    else:
        base_response["data"] = {
            "message": "Analysis completed with basic results",
            "confidence": 0.3
        }
    
    return base_response
