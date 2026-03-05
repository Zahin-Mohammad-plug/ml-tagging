"""Exception handling for the API"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import structlog
from typing import Dict, Any

logger = structlog.get_logger()

def handle_exceptions(app: FastAPI):
    """Add exception handlers to FastAPI app"""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        logger.warning(
            "HTTP exception",
            status_code=exc.status_code,
            detail=exc.detail,
            path=request.url.path
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "path": str(request.url.path)
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors"""
        logger.warning(
            "Validation error",
            errors=exc.errors(),
            path=request.url.path
        )
        
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation error",
                "status_code": 422,
                "details": exc.errors(),
                "path": str(request.url.path)
            }
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def starlette_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle Starlette HTTP exceptions"""
        logger.warning(
            "Starlette HTTP exception",
            status_code=exc.status_code,
            detail=exc.detail,
            path=request.url.path
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail or "HTTP Exception",
                "status_code": exc.status_code,
                "path": str(request.url.path)
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        logger.error(
            "Unhandled exception",
            error=str(exc),
            error_type=type(exc).__name__,
            path=request.url.path
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "status_code": 500,
                "path": str(request.url.path),
                "type": type(exc).__name__
            }
        )

class MLTaggerException(Exception):
    """Base exception for ML Tagger"""
    pass

class JobNotFoundException(MLTaggerException):
    """Job not found exception"""
    pass

class SuggestionNotFoundException(MLTaggerException):
    """Suggestion not found exception"""
    pass

class MLProcessingException(MLTaggerException):
    """ML processing exception"""
    pass