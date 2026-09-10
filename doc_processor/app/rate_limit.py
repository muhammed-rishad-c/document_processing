from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)


def key_func_by_api_key(request):
    
    return request.headers.get("x-api-key") or get_remote_address(request)


def key_func_by_session_id(request):
    
    return request.query_params.get("session_id") or get_remote_address(request)