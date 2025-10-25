#!/usr/bin/env python3
"""
Test script to reproduce the API response validation issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.schemas.analysis import FinalAnalysisResponse
from pydantic import ValidationError
import json

# This is the exact data structure returned by GraphExecutor (from logs)
test_data = {
    'media_id': '515ca2b3-f23b-4b34-a18b-0f726ce55ab5',
    'timestamp': '2025-08-15T21:59:26.164941',
    'overall_score': 5.3,
    'attractiveness_score': 5.6,
    'style_score': 5.0,
    'presence_score': 5.3,
    'summary': 'Your overall analysis shows a solid presentation with key strengths in Clear facial visibility.',
    'key_insights': ['Strong clear facial visibility', 'Detected 1 face(s) with good visibility'],
    'recommendations': [],
    'detailed_analysis': {
        'face': {
            'num_faces': 1,
            'faces': [{
                'bbox': [100, 50, 80, 80],
                'landmarks': {'left_eye': [110, 70], 'right_eye': [150, 70]},
                'crop_url': 'http://localhost:8000/storage/media/515ca2b3-f23b-4b34-a18b-0f726ce55ab5/OSK (1).jpg',
                'gender': None,
                'age': None,
                'age_range': None,
                'expression': None
            }]
        }
    },
    'confidence': 0.5,
    'processing_metadata': {
        'agents_used': ['embedding', 'fashion', 'posture', 'face'],
        'processing_mode': 'mock',
        'version': '1.0',
        'agent_successes': {
            'face': True,
            'fashion': False,
            'posture': False,
            'bio': False,
            'embedding': True
        },
        'processing_time': None,
        'mode': 'mock'
    },
    'langsmith_run_id': 'c8501399-96f6-4973-b4d8-cfc6a8041e65',
    'warnings': [
        'Analysis confidence is lower than usual - results may be less accurate.',
        'Some analyses were unavailable: fashion, posture, bio'
    ],
    'disclaimers': [
        'This analysis is for entertainment and self-improvement purposes only.',
        'Results are based on algorithmic analysis and should not be considered professional advice.'
    ]
}

print("Testing FinalAnalysisResponse validation...")
print("Data structure:")
print(json.dumps(test_data, indent=2))
print("\n" + "="*50 + "\n")

try:
    # Try to create FinalAnalysisResponse from the data
    response = FinalAnalysisResponse(**test_data)
    print("✅ SUCCESS: FinalAnalysisResponse validation passed!")
    print(f"Response: {response}")
except ValidationError as e:
    print("❌ VALIDATION ERROR:")
    print(e)
    print("\nDetailed errors:")
    for error in e.errors():
        print(f"  - Field: {error['loc']}, Error: {error['msg']}, Input: {error.get('input', 'N/A')}")
except Exception as e:
    print(f"❌ UNEXPECTED ERROR: {e}")