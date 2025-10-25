import time
from collections import defaultdict, deque
from typing import Dict, Deque, Tuple
from fastapi import HTTPException, Request
from functools import wraps
import asyncio

class InMemoryRateLimiter:
    """Simple in-memory rate limiter that doesn't require Redis"""
    
    def __init__(self):
        # Store request timestamps for each client
        self._requests: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """Check if request is allowed based on rate limit"""
        async with self._lock:
            now = time.time()
            window_start = now - window_seconds
            
            # Clean old requests outside the window
            requests = self._requests[key]
            while requests and requests[0] < window_start:
                requests.popleft()
            
            # Check if we're within the limit
            if len(requests) >= max_requests:
                return False
            
            # Add current request
            requests.append(now)
            return True
    
    def cleanup_old_entries(self):
        """Clean up old entries to prevent memory leaks"""
        now = time.time()
        # Remove entries older than 1 hour
        cutoff = now - 3600
        
        keys_to_remove = []
        for key, requests in self._requests.items():
            # Remove old requests
            while requests and requests[0] < cutoff:
                requests.popleft()
            
            # If no recent requests, mark key for removal
            if not requests:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._requests[key]

# Global rate limiter instance
_rate_limiter = InMemoryRateLimiter()

def get_client_key(request: Request) -> str:
    """Generate a unique key for the client"""
    # Use IP address as the key
    client_ip = request.client.host if request.client else "unknown"
    return f"ip:{client_ip}"

def rate_limit_dependency(max_requests: int, window_seconds: int):
    """Create a FastAPI dependency for rate limiting"""
    async def dependency(request: Request):
        client_key = get_client_key(request)
        
        if not await _rate_limiter.is_allowed(client_key, max_requests, window_seconds):
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {max_requests} requests per {window_seconds} seconds."
            )
        
        return None
    
    return dependency

# Predefined rate limiters
def rl_general_simple():
    """100 requests per minute"""
    return rate_limit_dependency(100, 60)

def rl_auth_simple():
    """10 requests per minute"""
    return rate_limit_dependency(10, 60)

def rl_upload_simple():
    """10 requests per hour"""
    return rate_limit_dependency(10, 3600)

# Background task to clean up old entries
async def cleanup_task():
    """Background task to clean up old rate limiter entries"""
    while True:
        await asyncio.sleep(300)  # Clean up every 5 minutes
        _rate_limiter.cleanup_old_entries()

# Start cleanup task
def start_cleanup_task():
    """Start the cleanup background task"""
    asyncio.create_task(cleanup_task())