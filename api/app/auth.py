"""Authentication utilities"""

from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from typing import Optional
import structlog

from .config import get_settings

logger = structlog.get_logger()
settings = get_settings()
security = HTTPBearer(auto_error=False)

# For now, we'll implement a simple API key authentication
# In production, consider implementing proper JWT with user management

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[str]:
    """
    Get current user from authentication credentials
    
    For now, this is a stub that allows anonymous access.
    In production, implement proper authentication here.
    """
    
    # Anonymous mode - no authentication required
    if not credentials:
        return "anonymous"
    
    # Simple API key validation (if provided)
    if credentials.credentials == settings.api_secret_key:
        return "api_user"
    
    # For JWT implementation (future):
    # try:
    #     payload = jwt.decode(
    #         credentials.credentials,
    #         settings.api_secret_key,
    #         algorithms=["HS256"]
    #     )
    #     username = payload.get("sub")
    #     if username is None:
    #         raise HTTPException(
    #             status_code=status.HTTP_401_UNAUTHORIZED,
    #             detail="Invalid authentication credentials",
    #         )
    #     return username
    # except JWTError:
    #     raise HTTPException(
    #         status_code=status.HTTP_401_UNAUTHORIZED,
    #         detail="Invalid authentication credentials",
    #     )
    
    # For now, allow any token
    return "authenticated_user"

def create_access_token(data: dict) -> str:
    """Create JWT access token (for future use)"""
    to_encode = data.copy()
    encoded_jwt = jwt.encode(to_encode, settings.api_secret_key, algorithm="HS256")
    return encoded_jwt

async def verify_api_key(api_key: str) -> bool:
    """Verify API key (simple implementation)"""
    return api_key == settings.api_secret_key