"""
Guardrails schemas for LLM output validation.
These schemas ensure that LLM outputs conform to expected formats and safety guidelines.
"""

from guardrails import Guard, OnFailAction
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Face Agent Guardrails Schema
class FaceAnalysisGuard(BaseModel):
    """Guardrails schema for Face Agent LLM outputs"""
    num_faces: int = Field(..., ge=0, le=10, description="Number of detected faces")
    faces: List[Dict[str, Any]] = Field(..., max_items=10, description="Face analysis results")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Analysis confidence")
    
    class Config:
        extra = "forbid"  # Prevent extra fields

# Fashion Agent Guardrails Schema  
class FashionAnalysisGuard(BaseModel):
    """Guardrails schema for Fashion Agent LLM outputs"""
    outfit_rating: int = Field(..., ge=0, le=100, description="Outfit rating 0-100")
    items: List[str] = Field(..., max_items=20, description="Detected clothing items")
    good: List[str] = Field(..., max_items=10, description="Positive aspects")
    bad: List[str] = Field(..., max_items=10, description="Areas for improvement")
    improvements: List[str] = Field(..., max_items=10, description="Specific suggestions")
    roast: Optional[str] = Field(None, max_length=500, description="Optional friendly roast")
    
    class Config:
        extra = "forbid"

# Bio Agent Guardrails Schema
class BioAnalysisGuard(BaseModel):
    """Guardrails schema for Bio Agent LLM outputs"""
    vibe_summary: str = Field(..., min_length=10, max_length=200, description="Brief vibe summary")
    strengths: List[str] = Field(..., min_items=1, max_items=8, description="Identified strengths")
    weaknesses: List[str] = Field(..., max_items=8, description="Areas for improvement")
    improvements: List[str] = Field(..., min_items=1, max_items=8, description="Specific suggestions")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Analysis confidence")
    
    class Config:
        extra = "forbid"

# Comparison Agent Guardrails Schema
class ComparisonGuard(BaseModel):
    """Guardrails schema for Comparison Agent LLM outputs"""
    similarities: List[str] = Field(..., max_items=8, description="Similar aspects")
    differences: List[str] = Field(..., max_items=8, description="Different aspects") 
    insights: List[str] = Field(..., min_items=1, max_items=8, description="Key insights")
    recommendations: List[str] = Field(..., max_items=8, description="Actionable recommendations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Comparison confidence")
    
    class Config:
        extra = "forbid"

# Safety and Content Filters
PROHIBITED_CONTENT_PATTERNS = [
    r"(?i)\b(suicide|self-harm|kill yourself|end it all)\b",
    r"(?i)\b(racist|sexist|homophobic|transphobic)\b", 
    r"(?i)\b(ugly|hideous|disgusting|repulsive)\b",
    r"(?i)\b(stupid|dumb|idiot|moron|retard)\b",
    r"(?i)\b(fat|obese|overweight)\s+(and|,)?\s+(ugly|disgusting)",
]

REQUIRED_DISCLAIMERS = [
    "for entertainment purposes only",
    "not professional advice", 
    "algorithmic analysis"
]

def create_fashion_guard() -> Guard:
    """Create Guardrails Guard for Fashion Agent outputs"""
    return Guard.from_pydantic(
        output_class=FashionAnalysisGuard,
        prompt="You are a helpful fashion advisor. Be constructive and avoid harsh criticism.",
        on_fail=OnFailAction.REASK
    )

def create_bio_guard() -> Guard:
    """Create Guardrails Guard for Bio Agent outputs"""
    return Guard.from_pydantic(
        output_class=BioAnalysisGuard,
        prompt="You are an empathetic communication advisor. Be supportive and constructive.",
        on_fail=OnFailAction.REASK
    )

def create_comparison_guard() -> Guard:
    """Create Guardrails Guard for Comparison Agent outputs"""
    return Guard.from_pydantic(
        output_class=ComparisonGuard,
        prompt="You are a comparison analyst. Be objective and constructive in your analysis.",
        on_fail=OnFailAction.REASK
    )

# Content safety validator
def validate_content_safety(text: str) -> tuple[bool, List[str]]:
    """
    Validate content for safety issues.
    Returns (is_safe, list_of_issues)
    """
    import re
    
    issues = []
    text_lower = text.lower()
    
    # Check for prohibited patterns
    for pattern in PROHIBITED_CONTENT_PATTERNS:
        if re.search(pattern, text):
            issues.append(f"Prohibited content detected: {pattern}")
    
    # Check for excessive negativity (simple heuristic)
    negative_words = ["bad", "terrible", "awful", "horrible", "disgusting", "ugly", "stupid"]
    negative_count = sum(1 for word in negative_words if word in text_lower)
    if negative_count > 3:
        issues.append("Excessive negative language detected")
    
    # Check for personal attacks
    attack_patterns = [
        r"you are (?:stupid|dumb|ugly|fat|disgusting)",
        r"you look (?:terrible|awful|disgusting|ugly)",
        r"(?:your|you're) (?:so|really) (?:stupid|dumb|ugly)"
    ]
    
    for pattern in attack_patterns:
        if re.search(pattern, text_lower):
            issues.append("Personal attack detected")
    
    return len(issues) == 0, issues

def sanitize_output(text: str) -> str:
    """
    Sanitize output by removing or replacing problematic content.
    """
    import re
    
    # Replace harsh words with milder alternatives
    replacements = {
        r"\bugly\b": "less flattering",
        r"\bstupid\b": "less effective",
        r"\bdumb\b": "could be improved", 
        r"\bterrible\b": "needs work",
        r"\bawful\b": "could be better",
        r"\bhideous\b": "unflattering",
        r"\bdisgust(ing|ed)\b": "not ideal"
    }
    
    sanitized = text
    for pattern, replacement in replacements.items():
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    
    return sanitized

# Validation decorators for agents
def validate_fashion_output(func):
    """Decorator to validate Fashion Agent outputs"""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        if not result.get("success"):
            return result
            
        try:
            # Create guard and validate
            guard = create_fashion_guard()
            validated_data = guard.parse(result["data"])
            
            # Additional safety check
            output_text = str(result["data"])
            is_safe, issues = validate_content_safety(output_text)
            
            if not is_safe:
                # Sanitize the output
                if "roast" in result["data"]:
                    result["data"]["roast"] = sanitize_output(result["data"]["roast"])
                if "improvements" in result["data"]:
                    result["data"]["improvements"] = [
                        sanitize_output(imp) for imp in result["data"]["improvements"]
                    ]
                
                result["warnings"] = result.get("warnings", []) + ["Content was sanitized for safety"]
            
            return result
            
        except Exception as e:
            # If validation fails, return sanitized fallback
            return {
                "success": True,
                "data": {
                    "outfit_rating": 50,
                    "items": result["data"].get("items", []),
                    "good": ["Shows personal style"],
                    "bad": [],
                    "improvements": ["Consider experimenting with different styles"],
                    "roast": None
                },
                "warnings": [f"Output sanitized due to validation error: {str(e)}"]
            }
    
    return wrapper

def validate_bio_output(func):
    """Decorator to validate Bio Agent outputs"""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        if not result.get("success"):
            return result
            
        try:
            # Create guard and validate
            guard = create_bio_guard()
            validated_data = guard.parse(result["data"])
            
            # Safety check
            output_text = str(result["data"])
            is_safe, issues = validate_content_safety(output_text)
            
            if not is_safe:
                # Sanitize problematic content
                for field in ["vibe_summary", "strengths", "weaknesses", "improvements"]:
                    if field in result["data"]:
                        if isinstance(result["data"][field], str):
                            result["data"][field] = sanitize_output(result["data"][field])
                        elif isinstance(result["data"][field], list):
                            result["data"][field] = [
                                sanitize_output(item) for item in result["data"][field]
                            ]
                
                result["warnings"] = result.get("warnings", []) + ["Content was sanitized for safety"]
            
            return result
            
        except Exception as e:
            # Fallback response
            return {
                "success": True,
                "data": {
                    "vibe_summary": "Shows authentic personality",
                    "strengths": ["Clear communication"],
                    "weaknesses": ["Could be more detailed"],
                    "improvements": ["Consider adding specific examples"],
                    "confidence": 0.5
                },
                "warnings": [f"Output sanitized due to validation error: {str(e)}"]
            }
    
    return wrapper
