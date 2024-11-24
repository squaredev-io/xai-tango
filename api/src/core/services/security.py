from fastapi import HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette import status
from api.src.core.services.settings import get_settings
import os 

settings = get_settings()

# Schema details for API Key
api_key_schema_name = "API Key Authentication"
api_key_description = """
This API uses API Key authentication. To authenticate, add your API Key to the `X-API-KEY` header of your request.
"""

# Schema details for Token
token_schema_name = "Token Authentication"
token_description = """
This API uses Token authentication. To authenticate, add your Bearer Token to the `Authorization` header of your request.
"""

# Define the API Key header
api_key_header = APIKeyHeader(
    name=settings.API_KEY_NAME,
    scheme_name=api_key_schema_name,
    description=api_key_description,
    auto_error=False,
)



def get_api_key(
    api_key_header: str = Security(api_key_header),
):
    """
    Retrieve & validate either API Key or Bearer Token.
    """
    # Check API Key
    if api_key_header:
        if api_key_header not in settings.API_KEYS:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API Key",
            )
        return {"auth_type": "api_key", "auth_value": api_key_header}


    # If neither API Key nor Token is provided
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key ",
    )


