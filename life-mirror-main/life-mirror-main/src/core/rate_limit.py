import os
from fastapi import Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis.asyncio as redis
import fakeredis.aioredis
from .simple_rate_limiter import (
    rl_general_simple, rl_auth_simple, rl_upload_simple, start_cleanup_task
)

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
USE_FAKE_REDIS = os.getenv("USE_FAKE_REDIS", "true").lower() == "true"
USE_SIMPLE_RATE_LIMITER = os.getenv("USE_SIMPLE_RATE_LIMITER", "false").lower() == "true"

# Global state
_redis_rate_limiting_enabled = False
limiter = None

async def init_rate_limiter(app=None):
    global _redis_rate_limiting_enabled, limiter
    
    # If simple rate limiter is preferred, use it
    if USE_SIMPLE_RATE_LIMITER:
        print("Using simple in-memory rate limiter (development mode)")
        start_cleanup_task()
        return
    
    # Try to initialize Redis-based rate limiter with slowapi
    try:
        if USE_FAKE_REDIS:
            # Use in-memory storage for development
            limiter = Limiter(
                key_func=get_remote_address,
                storage_uri="memory://"
            )
            print("Using slowapi with in-memory storage for rate limiting (development mode)")
        else:
            # Use real Redis
            limiter = Limiter(
                key_func=get_remote_address,
                storage_uri=REDIS_URL
            )
            print("Using slowapi with Redis for rate limiting")
        
        # Set the limiter in app state if app is provided
        if app:
            app.state.limiter = limiter
        
        _redis_rate_limiting_enabled = True
        print("Rate limiter initialized successfully with slowapi")
    except Exception as e:
        print(f"Failed to initialize slowapi rate limiter: {e}")
        print("Falling back to simple in-memory rate limiter")
        start_cleanup_task()

def _no_op_dependency():
    """No-op dependency when rate limiting is disabled"""
    return None

def rl_general():
    if _redis_rate_limiting_enabled and limiter:
        return limiter.limit("100/minute")  # 100 requests per minute
    return rl_general_simple()

def rl_auth():
    if _redis_rate_limiting_enabled and limiter:
        return limiter.limit("2/minute")   # 2 requests per minute for testing
    return rl_auth_simple()

def rl_upload():
    if _redis_rate_limiting_enabled and limiter:
        return limiter.limit("10/hour")     # 10 requests per hour
    return rl_upload_simple()

def rate_limit(endpoint: str, max_calls: int, window_seconds: int):
    """Rate limiting decorator"""
    def decorator(func):
        return func  # Simple passthrough for now
    return decorator
