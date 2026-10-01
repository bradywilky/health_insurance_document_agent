"""Who is asking, for audit records.

Behind an authenticating proxy (an ALB with OIDC/Cognito, or an identity-aware proxy) the proxy sets the user in a
request header. Those headers are trusted only when AUDIT_TRUST_HEADERS=true, because without such a proxy any
client can send them. Otherwise AUDIT_USER, then the operating-system user, are recorded as unauthenticated.
"""
import getpass
import os

# Checked in order; the first present header wins.
IDENTITY_HEADERS = ('X-Amzn-Oidc-Identity', 'X-Forwarded-User', 'X-Forwarded-Email', 'X-Auth-Request-Email',
                    'X-Auth-Request-User', 'X-Remote-User')


def _get(headers, name):
    if not headers:
        return None
    for key in (name, name.lower()):
        try:
            value = headers.get(key)
        except Exception:
            value = None
        if value:
            return str(value).strip()[:256]
    return None


def resolve_user(headers=None):
    if os.getenv('AUDIT_TRUST_HEADERS', 'false').strip().lower() == 'true':
        for name in IDENTITY_HEADERS:
            value = _get(headers, name)
            if value:
                return {'id': value, 'source': f'header:{name}', 'authenticated': True}
    if os.getenv('AUDIT_USER'):
        return {'id': os.getenv('AUDIT_USER'), 'source': 'env:AUDIT_USER', 'authenticated': False}
    try:
        return {'id': getpass.getuser(), 'source': 'os', 'authenticated': False}
    except Exception:
        return {'id': 'unknown', 'source': 'none', 'authenticated': False}
