"""GitHub App JWT creation for installation authentication."""

import time
from pathlib import Path

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


class GitHubAppAuthenticator:
    """Create signed GitHub App credentials from the configured private key."""

    def __init__(self, app_id: str, private_key_path: Path) -> None:
        self.app_id = app_id
        self.private_key_path = private_key_path

    def app_jwt(self) -> str:
        """Sign and return a short-lived JWT for GitHub App authentication."""
        private_key = serialization.load_pem_private_key(
            self.private_key_path.read_bytes(), password=None
        )
        if not isinstance(private_key, rsa.RSAPrivateKey):
            raise ValueError("GitHub App private key must be an RSA key")
        now = int(time.time())
        return jwt.encode(
            {"iat": now - 30, "exp": now + 540, "iss": self.app_id},
            private_key,
            algorithm="RS256",
        )
