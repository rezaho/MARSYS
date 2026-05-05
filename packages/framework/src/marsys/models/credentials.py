"""
OAuth Credential Store for Multi-Account Management.

Provides centralized management of OAuth credentials for multiple accounts
across different providers (anthropic-oauth, openai-oauth) with automatic token
refresh support.

Example usage:
    # Get credential store singleton
    store = OAuthCredentialStore.get_instance()

    # Add a profile
    store.add_profile(OAuthProfile(
        name="work-claude",
        provider="anthropic-oauth",
        credentials_path="~/.claude/.credentials.json"
    ))

    # Set as default for provider
    store.set_default("anthropic-oauth", "work-claude")

    # Resolve credentials path for a profile
    path = store.resolve_credentials_path("work-claude", "anthropic-oauth")
"""

import json
import logging
import os
import stat
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Supported OAuth providers
SUPPORTED_PROVIDERS = {"anthropic-oauth", "openai-oauth"}

# Default credential file paths for auto-discovery
DEFAULT_CREDENTIAL_PATHS = {
    "anthropic-oauth": "~/.claude/.credentials.json",
    "openai-oauth": "~/.codex/auth.json",
}

# OAuth refresh endpoints and client IDs
OAUTH_CONFIG = {
    "anthropic-oauth": {
        "token_endpoint": "https://console.anthropic.com/v1/oauth/token",
        "client_id": "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
        "refresh_buffer_seconds": 300,  # 5 minutes
    },
    "openai-oauth": {
        "token_endpoint": "https://auth.openai.com/oauth/token",
        "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
        "refresh_buffer_seconds": 300,  # 5 minutes
    },
}

# Default store location
DEFAULT_STORE_PATH = "~/.marsys/credentials.json"


# =============================================================================
# OAuthProfile
# =============================================================================

@dataclass
class OAuthProfile:
    """
    Represents a single OAuth credential profile.

    A profile maps a friendly name to a credentials file for a specific provider.
    """

    name: str
    provider: str
    credentials_path: str
    description: Optional[str] = None
    created_at: Optional[str] = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_used_at: Optional[str] = None

    def __post_init__(self):
        """Validate profile after initialization."""
        if self.provider not in SUPPORTED_PROVIDERS:
            from marsys.agents.exceptions import AgentConfigurationError
            raise AgentConfigurationError(
                f"Invalid OAuth provider: {self.provider}",
                config_field="provider",
                config_value=self.provider,
                suggestion=f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
            )

    @property
    def resolved_path(self) -> Path:
        """Get the fully resolved credentials path (expands ~)."""
        return Path(self.credentials_path).expanduser()

    def exists(self) -> bool:
        """Check if the credentials file exists."""
        return self.resolved_path.exists()

    def is_valid_format(self) -> bool:
        """
        Check if the credentials file has valid JSON format.

        Does not validate the actual token content, just JSON structure.
        """
        if not self.exists():
            return False

        try:
            with open(self.resolved_path) as f:
                data = json.load(f)

            # Provider-specific format validation
            if self.provider == "anthropic-oauth":
                return "claudeAiOauth" in data and "accessToken" in data.get("claudeAiOauth", {})
            elif self.provider == "openai-oauth":
                tokens = data.get("tokens", {})
                return "access_token" in tokens
            return False
        except (json.JSONDecodeError, OSError):
            return False

    def validate(self) -> None:
        """
        Validate the profile's credentials file.

        Raises:
            AgentConfigurationError: If validation fails
        """
        from marsys.agents.exceptions import AgentConfigurationError

        if not self.exists():
            raise AgentConfigurationError(
                f"Credentials file not found for profile '{self.name}'",
                config_field="credentials_path",
                config_value=str(self.resolved_path),
                suggestion=f"Run 'marsys oauth add {self.name} --provider {self.provider}' to authenticate"
            )

        if not self.is_valid_format():
            raise AgentConfigurationError(
                f"Invalid credentials format for profile '{self.name}'",
                config_field="credentials_path",
                config_value=str(self.resolved_path),
                suggestion="Credentials file may be corrupted. Try re-authenticating."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize profile to dictionary."""
        return {
            "provider": self.provider,
            "credentials_path": self.credentials_path,
            "description": self.description,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
        }

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "OAuthProfile":
        """Deserialize profile from dictionary."""
        return cls(
            name=name,
            provider=data["provider"],
            credentials_path=data["credentials_path"],
            description=data.get("description"),
            created_at=data.get("created_at"),
            last_used_at=data.get("last_used_at"),
        )


# =============================================================================
# OAuthTokenRefresher
# =============================================================================

class OAuthTokenRefresher:
    """
    Utility class for refreshing OAuth tokens.

    Handles token refresh for both Anthropic and OpenAI OAuth providers.
    """

    @classmethod
    def refresh_if_needed(
        cls,
        credentials_path: str,
        provider: str,
        force: bool = False
    ) -> bool:
        """
        Refresh OAuth token if expiring soon or if forced.

        Args:
            credentials_path: Path to credentials file
            provider: OAuth provider name
            force: Force refresh even if not expired

        Returns:
            True if refresh was performed, False otherwise

        Raises:
            AgentConfigurationError: If refresh fails
        """
        if provider not in OAUTH_CONFIG:
            logger.debug(f"No refresh config for provider {provider}")
            return False

        cred_path = Path(credentials_path).expanduser()
        if not cred_path.exists():
            return False

        try:
            with open(cred_path) as f:
                data = json.load(f)

            # Extract tokens based on provider
            if provider == "anthropic-oauth":
                oauth_data = data.get("claudeAiOauth", {})
                refresh_token = oauth_data.get("refreshToken")
                expires_at = oauth_data.get("expiresAt")  # Unix timestamp in milliseconds
            elif provider == "openai-oauth":
                tokens = data.get("tokens", {})
                refresh_token = tokens.get("refresh_token")
                # OpenAI uses JWT, expiration is in the token itself
                expires_at = cls._get_jwt_expiration(tokens.get("access_token"))
            else:
                return False

            if not refresh_token:
                logger.debug(f"No refresh token available for {provider}")
                return False

            # Check if refresh is needed
            config = OAUTH_CONFIG[provider]
            buffer_seconds = config["refresh_buffer_seconds"]

            if expires_at and not force:
                # Convert to seconds if in milliseconds
                expires_at_sec = expires_at / 1000 if expires_at > 10**12 else expires_at
                if time.time() < (expires_at_sec - buffer_seconds):
                    logger.debug(f"Token still valid for {provider}, no refresh needed")
                    return False

            # Perform refresh
            logger.info(f"Refreshing OAuth token for {provider}")
            new_tokens = cls._call_refresh_endpoint(provider, refresh_token)

            # Update credentials file
            cls._update_credential_file(cred_path, provider, new_tokens)

            logger.info(f"Successfully refreshed OAuth token for {provider}")
            return True

        except Exception as e:
            from marsys.agents.exceptions import AgentConfigurationError

            # Check if it's an invalid_grant error (refresh token revoked/expired)
            if "invalid_grant" in str(e).lower():
                cli_cmds = {
                    "anthropic-oauth": "claude login",
                    "openai-oauth": "codex login",
                }
                cli_cmd = cli_cmds.get(provider, f"re-authenticate with {provider}")
                raise AgentConfigurationError(
                    f"OAuth refresh token expired or revoked for {provider}",
                    config_field="refresh_token",
                    suggestion=f"Run '{cli_cmd}' to re-authenticate"
                )

            logger.warning(f"Failed to refresh token for {provider}: {e}")
            return False

    @staticmethod
    def _get_jwt_expiration(access_token: Optional[str]) -> Optional[int]:
        """Extract expiration timestamp from JWT token."""
        if not access_token:
            return None

        try:
            import base64

            # JWT has 3 parts: header.payload.signature
            payload_b64 = access_token.split(".")[1]
            # Add padding for base64 decoding
            payload_b64 += "=" * (4 - len(payload_b64) % 4)
            payload = json.loads(base64.b64decode(payload_b64))

            return payload.get("exp")
        except Exception:
            return None

    @staticmethod
    def _call_refresh_endpoint(provider: str, refresh_token: str) -> Dict[str, Any]:
        """
        Call OAuth refresh endpoint.

        Returns:
            Dictionary with new tokens

        Raises:
            Exception on failure
        """
        config = OAUTH_CONFIG[provider]

        data = {
            "grant_type": "refresh_token",
            "client_id": config["client_id"],
            "refresh_token": refresh_token,
        }
        if "client_secret" in config:
            data["client_secret"] = config["client_secret"]

        response = requests.post(
            config["token_endpoint"],
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )

        if response.status_code != 200:
            raise Exception(f"Token refresh failed: {response.status_code} - {response.text}")

        return response.json()

    @staticmethod
    def _update_credential_file(
        cred_path: Path,
        provider: str,
        new_tokens: Dict[str, Any]
    ) -> None:
        """Update credential file with new tokens.

        Uses file locking to prevent concurrent refresh corruption when
        multiple agents share the same credential file.
        """
        import fcntl
        import tempfile

        lock_path = Path(f"{cred_path}.lock")
        lock_path.touch(exist_ok=True)

        with open(lock_path) as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                with open(cred_path) as f:
                    data = json.load(f)

                current_time_ms = int(time.time() * 1000)

                if provider == "anthropic-oauth":
                    oauth_data = data.setdefault("claudeAiOauth", {})
                    oauth_data["accessToken"] = new_tokens["access_token"]
                    expires_in = new_tokens.get("expires_in", 3600)
                    oauth_data["expiresAt"] = current_time_ms + (expires_in * 1000)
                    if "refresh_token" in new_tokens:
                        oauth_data["refreshToken"] = new_tokens["refresh_token"]

                elif provider == "openai-oauth":
                    tokens = data.setdefault("tokens", {})
                    tokens["access_token"] = new_tokens["access_token"]
                    if "refresh_token" in new_tokens:
                        tokens["refresh_token"] = new_tokens["refresh_token"]

                # Atomic write: write to temp file then rename
                dir_path = cred_path.parent
                fd, tmp_path = tempfile.mkstemp(dir=str(dir_path), suffix=".tmp")
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(data, f, indent=2)
                    os.chmod(tmp_path, stat.S_IRUSR | stat.S_IWUSR)
                    os.replace(tmp_path, str(cred_path))
                except Exception:
                    # Clean up temp file on failure
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)


# =============================================================================
# OAuthCredentialStore
# =============================================================================

class OAuthCredentialStore:
    """
    Singleton store for managing OAuth credential profiles.

    Provides centralized management of multiple OAuth accounts across providers,
    with support for profile defaults and auto-discovery.
    """

    _instance: Optional["OAuthCredentialStore"] = None
    _lock = Lock()

    def __init__(self, store_path: Optional[str] = None):
        """
        Initialize credential store.

        Args:
            store_path: Path to store config (default: ~/.marsys/credentials.json)
        """
        self._store_path = Path(
            store_path or os.getenv("MARSYS_CREDENTIALS_PATH", DEFAULT_STORE_PATH)
        ).expanduser()

        self._profiles: Dict[str, OAuthProfile] = {}
        self._defaults: Dict[str, str] = {}
        self._version: int = 1

        # Load existing config or auto-discover
        if self._store_path.exists():
            self._load()
        else:
            self._auto_discover()

    @classmethod
    def get_instance(cls, store_path: Optional[str] = None) -> "OAuthCredentialStore":
        """Get singleton instance of credential store."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(store_path)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        with cls._lock:
            cls._instance = None

    # -------------------------------------------------------------------------
    # Profile Management
    # -------------------------------------------------------------------------

    def add_profile(self, profile: OAuthProfile, set_as_default: bool = False) -> None:
        """
        Add or update a credential profile.

        Args:
            profile: The profile to add
            set_as_default: If True, set this profile as default for its provider
        """
        self._profiles[profile.name] = profile

        if set_as_default:
            self._defaults[profile.provider] = profile.name

        logger.info(f"Added OAuth profile: {profile.name} ({profile.provider})")

    def remove_profile(self, name: str) -> bool:
        """
        Remove a credential profile.

        Args:
            name: Profile name to remove

        Returns:
            True if profile was removed, False if not found
        """
        if name not in self._profiles:
            return False

        profile = self._profiles.pop(name)

        # Remove from defaults if it was the default
        if self._defaults.get(profile.provider) == name:
            del self._defaults[profile.provider]

        logger.info(f"Removed OAuth profile: {name}")
        return True

    def get_profile(self, name: str) -> Optional[OAuthProfile]:
        """Get a profile by name."""
        return self._profiles.get(name)

    def list_profiles(self, provider: Optional[str] = None) -> List[OAuthProfile]:
        """
        List all profiles, optionally filtered by provider.

        Args:
            provider: Optional provider to filter by

        Returns:
            List of matching profiles
        """
        profiles = list(self._profiles.values())

        if provider:
            profiles = [p for p in profiles if p.provider == provider]

        return profiles

    # -------------------------------------------------------------------------
    # Default Management
    # -------------------------------------------------------------------------

    def set_default(self, provider: str, profile_name: str) -> None:
        """
        Set the default profile for a provider.

        Args:
            provider: Provider name
            profile_name: Profile name to set as default

        Raises:
            AgentConfigurationError: If profile doesn't exist or doesn't match provider
        """
        from marsys.agents.exceptions import AgentConfigurationError

        profile = self._profiles.get(profile_name)
        if not profile:
            raise AgentConfigurationError(
                f"Profile '{profile_name}' not found",
                config_field="profile_name",
                suggestion=f"Use 'marsys oauth list' to see available profiles"
            )

        if profile.provider != provider:
            raise AgentConfigurationError(
                f"Profile '{profile_name}' is for provider '{profile.provider}', not '{provider}'",
                config_field="provider",
                suggestion=f"Choose a profile with provider '{provider}'"
            )

        self._defaults[provider] = profile_name
        logger.info(f"Set default profile for {provider}: {profile_name}")

    def get_default(self, provider: str) -> Optional[str]:
        """Get the default profile name for a provider."""
        return self._defaults.get(provider)

    def get_default_profile(self, provider: str) -> Optional[OAuthProfile]:
        """Get the default profile object for a provider."""
        default_name = self.get_default(provider)
        if default_name:
            return self._profiles.get(default_name)
        return None

    # -------------------------------------------------------------------------
    # Credential Resolution
    # -------------------------------------------------------------------------

    def resolve_credentials_path(
        self,
        oauth_profile: Optional[str],
        provider: str,
        auto_refresh: bool = True
    ) -> Optional[str]:
        """
        Resolve OAuth profile to credentials file path.

        Resolution order:
        1. If oauth_profile specified, use that profile
        2. Else use default profile for provider (if set)
        3. Else return None (caller should use provider default)

        Args:
            oauth_profile: Profile name to resolve
            provider: Provider name for validation
            auto_refresh: If True, attempt to refresh token if expiring

        Returns:
            Resolved credentials path or None

        Raises:
            AgentConfigurationError: If profile doesn't exist or is invalid
        """
        from marsys.agents.exceptions import AgentConfigurationError

        profile: Optional[OAuthProfile] = None

        # Try to resolve profile
        if oauth_profile:
            profile = self._profiles.get(oauth_profile)
            if not profile:
                raise AgentConfigurationError(
                    f"OAuth profile '{oauth_profile}' not found",
                    config_field="oauth_profile",
                    config_value=oauth_profile,
                    suggestion=f"Use 'marsys oauth list' to see available profiles, or "
                               f"'marsys oauth add {oauth_profile} --provider {provider}' to create one"
                )

            if profile.provider != provider:
                raise AgentConfigurationError(
                    f"OAuth profile '{oauth_profile}' is for provider '{profile.provider}', "
                    f"not '{provider}'",
                    config_field="oauth_profile",
                    config_value=oauth_profile,
                    suggestion=f"Use a profile with provider '{provider}'"
                )
        else:
            # Try default profile
            profile = self.get_default_profile(provider)

        if not profile:
            return None

        # Validate profile
        profile.validate()

        # Update last_used timestamp
        profile.last_used_at = datetime.now(timezone.utc).isoformat()

        # Auto-refresh if enabled
        if auto_refresh:
            try:
                OAuthTokenRefresher.refresh_if_needed(
                    str(profile.resolved_path),
                    profile.provider
                )
            except Exception as e:
                logger.warning(f"Auto-refresh failed for profile '{profile.name}': {e}")

        return str(profile.resolved_path)

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self) -> None:
        """
        Save store configuration to disk.

        Creates parent directory if needed and sets secure file permissions.
        """
        # Ensure parent directory exists
        self._store_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": self._version,
            "profiles": {
                name: profile.to_dict()
                for name, profile in self._profiles.items()
            },
            "defaults": self._defaults,
        }

        with open(self._store_path, "w") as f:
            json.dump(data, f, indent=2)

        # Set secure file permissions (owner read/write only)
        os.chmod(self._store_path, stat.S_IRUSR | stat.S_IWUSR)

        logger.debug(f"Saved credential store to {self._store_path}")

    def _load(self) -> None:
        """Load store configuration from disk."""
        try:
            # Warn if file permissions are too open
            file_stat = os.stat(self._store_path)
            mode = file_stat.st_mode
            if mode & (stat.S_IRGRP | stat.S_IROTH):
                logger.warning(
                    f"Credentials store at {self._store_path} has overly permissive permissions. "
                    f"Consider running: chmod 600 {self._store_path}"
                )

            with open(self._store_path) as f:
                data = json.load(f)

            self._version = data.get("version", 1)
            self._defaults = data.get("defaults", {})

            # Load profiles
            for name, profile_data in data.get("profiles", {}).items():
                try:
                    self._profiles[name] = OAuthProfile.from_dict(name, profile_data)
                except Exception as e:
                    logger.warning(f"Failed to load profile '{name}': {e}")

            logger.debug(f"Loaded {len(self._profiles)} profiles from {self._store_path}")

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse credentials store: {e}")
        except Exception as e:
            logger.warning(f"Failed to load credentials store: {e}")

    def _auto_discover(self) -> None:
        """
        Auto-discover existing credential files from CLI tools.

        Checks default locations for Claude CLI and Codex CLI credentials
        and creates profiles for any found.
        """
        discovered = 0

        for provider, default_path in DEFAULT_CREDENTIAL_PATHS.items():
            paths_to_check = [default_path]

            for check_path in paths_to_check:
                cred_path = Path(check_path).expanduser()

                if cred_path.exists():
                    profile_name = f"default-{provider.replace('-oauth', '')}"

                    try:
                        profile = OAuthProfile(
                            name=profile_name,
                            provider=provider,
                            credentials_path=check_path,
                            description=f"Auto-discovered from {check_path}",
                        )

                        if profile.is_valid_format():
                            self._profiles[profile_name] = profile
                            self._defaults[provider] = profile_name
                            discovered += 1
                            logger.info(f"Auto-discovered OAuth credentials: {profile_name} ({cred_path})")
                            break  # Found valid credentials, skip secondary paths
                    except Exception as e:
                        logger.debug(f"Failed to auto-discover {provider}: {e}")

        if discovered > 0:
            # Save the discovered profiles
            self.save()


# =============================================================================
# Credential Update Helpers
# =============================================================================

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "OAuthProfile",
    "OAuthCredentialStore",
    "OAuthTokenRefresher",
    "SUPPORTED_PROVIDERS",
    "DEFAULT_CREDENTIAL_PATHS",
    "OAUTH_CONFIG",
]
