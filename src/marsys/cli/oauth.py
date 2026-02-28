"""
OAuth credential management CLI commands.

Provides commands for managing OAuth credential profiles:
- add: Add a new OAuth profile via browser login
- list: List all OAuth profiles
- remove: Remove an OAuth profile
- set-default: Set the default profile for a provider
- refresh: Manually refresh a profile's OAuth token
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click

from marsys.models.credentials import (
    OAuthCredentialStore,
    OAuthProfile,
    OAuthTokenRefresher,
    SUPPORTED_PROVIDERS,
)

_PROVIDER_CHOICES = ["anthropic-oauth", "openai-oauth"]


@click.group()
def oauth():
    """Manage OAuth credential profiles.

    OAuth profiles allow you to use multiple accounts for Claude Max (anthropic-oauth)
    and ChatGPT/Codex (openai-oauth) subscriptions.

    \b
    Examples:
        marsys oauth list
        marsys oauth add work-claude --provider anthropic-oauth
        marsys oauth set-default anthropic-oauth work-claude
        marsys oauth refresh work-claude
    """
    pass


@oauth.command()
@click.argument("profile_name")
@click.option(
    "--provider",
    type=click.Choice(_PROVIDER_CHOICES),
    required=True,
    help="OAuth provider for this profile"
)
@click.option(
    "--set-default",
    is_flag=True,
    help="Set this profile as the default for its provider"
)
@click.option(
    "--description",
    default=None,
    help="Optional description for this profile"
)
def add(profile_name: str, provider: str, set_default: bool,
        description: Optional[str]):
    """Add a new OAuth profile via browser login.

    This command will:
    1. Create a profile directory at ~/.marsys/oauth/<profile-name>/
    2. Launch the appropriate authentication flow
    3. Register the profile in the credential store

    \b
    Examples:
        marsys oauth add work-claude --provider anthropic-oauth
        marsys oauth add team-openai --provider openai-oauth --set-default
    """
    store = OAuthCredentialStore.get_instance()

    # Check if profile already exists
    if store.get_profile(profile_name):
        click.echo(f"Error: Profile '{profile_name}' already exists.", err=True)
        click.echo("Use 'marsys oauth remove' first, or choose a different name.", err=True)
        sys.exit(1)

    # Determine CLI tool and credential file name
    if provider == "anthropic-oauth":
        cli_tool = "claude"
        cli_login_cmd = "claude login"
        env_var = "CLAUDE_CONFIG_DIR"
        cred_file = ".credentials.json"
    elif provider == "openai-oauth":
        cli_tool = "codex"
        cli_login_cmd = "codex login"
        env_var = "CODEX_HOME"  # May vary, check codex docs
        cred_file = "auth.json"
    else:
        click.echo(f"Error: Unsupported provider '{provider}'", err=True)
        sys.exit(1)

    # Check if CLI tool is installed
    if not _is_command_available(cli_tool):
        click.echo(f"Error: '{cli_tool}' CLI is not installed.", err=True)
        if provider == "anthropic-oauth":
            click.echo("Install it with: npm install -g @anthropic-ai/claude-cli", err=True)
        else:
            click.echo("Install it with: npm install -g @openai/codex-cli", err=True)
        sys.exit(1)

    # Create profile directory
    profile_dir = Path.home() / ".marsys" / "oauth" / profile_name
    profile_dir.mkdir(parents=True, exist_ok=True)

    credentials_path = profile_dir / cred_file

    click.echo(f"\nCreating OAuth profile: {profile_name}")
    click.echo(f"Provider: {provider}")
    click.echo(f"Credentials will be stored at: {credentials_path}\n")

    # Set environment variable to redirect credential storage
    env = os.environ.copy()
    env[env_var] = str(profile_dir)

    click.echo(f"Launching '{cli_login_cmd}' for authentication...")
    click.echo("Please complete the login in your browser.\n")

    try:
        # Run the login command
        result = subprocess.run(
            [cli_tool, "login"],
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        click.echo(f"\nError: Login failed with exit code {e.returncode}", err=True)
        sys.exit(1)
    except FileNotFoundError:
        click.echo(f"\nError: '{cli_tool}' command not found", err=True)
        sys.exit(1)

    # Verify credentials were created
    if not credentials_path.exists():
        click.echo(f"\nError: Credentials file was not created at {credentials_path}", err=True)
        click.echo("The login may have failed or the CLI stores credentials elsewhere.", err=True)
        sys.exit(1)

    # Create and register profile
    profile = OAuthProfile(
        name=profile_name,
        provider=provider,
        credentials_path=str(credentials_path),
        description=description or f"Added via CLI",
    )

    store.add_profile(profile, set_as_default=set_default)
    store.save()

    click.echo(f"\nProfile '{profile_name}' created successfully!")
    if set_default:
        click.echo(f"Set as default for {provider}")


@oauth.command("list")
@click.option(
    "--provider",
    type=click.Choice(_PROVIDER_CHOICES),
    default=None,
    help="Filter by provider"
)
def list_profiles(provider: Optional[str]):
    """List all OAuth profiles.

    Shows profile name, provider, credentials path, and whether it's the default.

    \b
    Examples:
        marsys oauth list
        marsys oauth list --provider anthropic-oauth
    """
    store = OAuthCredentialStore.get_instance()
    profiles = store.list_profiles(provider)

    if not profiles:
        if provider:
            click.echo(f"No profiles found for provider '{provider}'")
        else:
            click.echo("No OAuth profiles configured.")
        click.echo("\nUse 'marsys oauth add <name> --provider <provider>' to add one.")
        return

    # Get defaults for marking
    defaults = {p: store.get_default(p) for p in SUPPORTED_PROVIDERS}

    # Header
    click.echo(f"\n{'Profile':<20} {'Provider':<18} {'Path':<45} {'Status':<10}")
    click.echo("-" * 95)

    for profile in profiles:
        is_default = defaults.get(profile.provider) == profile.name
        status = "default" if is_default else ""
        if profile.exists():
            if not profile.is_valid_format():
                status = "invalid"
        else:
            status = "missing"

        path_str = profile.credentials_path
        if len(path_str) > 43:
            path_str = "..." + path_str[-40:]

        click.echo(f"{profile.name:<20} {profile.provider:<18} {path_str:<45} {status:<10}")

    click.echo()


@oauth.command()
@click.argument("profile_name")
@click.option(
    "--force", "-f",
    is_flag=True,
    help="Remove without confirmation"
)
def remove(profile_name: str, force: bool):
    """Remove an OAuth profile.

    This removes the profile from the credential store but does NOT delete
    the credentials file. You can re-add the profile later.

    \b
    Examples:
        marsys oauth remove work-claude
        marsys oauth remove work-claude --force
    """
    store = OAuthCredentialStore.get_instance()
    profile = store.get_profile(profile_name)

    if not profile:
        click.echo(f"Error: Profile '{profile_name}' not found.", err=True)
        sys.exit(1)

    if not force:
        click.confirm(
            f"Remove profile '{profile_name}'? (Credentials file will not be deleted)",
            abort=True
        )

    store.remove_profile(profile_name)
    store.save()

    click.echo(f"Profile '{profile_name}' removed.")


@oauth.command("set-default")
@click.argument("provider", type=click.Choice(_PROVIDER_CHOICES))
@click.argument("profile_name")
def set_default(provider: str, profile_name: str):
    """Set the default profile for a provider.

    The default profile is used when no oauth_profile is specified in the
    ModelConfig.

    \b
    Examples:
        marsys oauth set-default anthropic-oauth work-claude
        marsys oauth set-default openai-oauth team-openai
    """
    store = OAuthCredentialStore.get_instance()

    try:
        store.set_default(provider, profile_name)
        store.save()
        click.echo(f"Set '{profile_name}' as default for {provider}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@oauth.command()
@click.argument("profile_name")
@click.option(
    "--force", "-f",
    is_flag=True,
    help="Force refresh even if token is not expiring"
)
def refresh(profile_name: str, force: bool):
    """Manually refresh a profile's OAuth token.

    Tokens are normally refreshed automatically before expiration.
    Use this command to manually trigger a refresh.

    \b
    Examples:
        marsys oauth refresh work-claude
        marsys oauth refresh work-claude --force
    """
    store = OAuthCredentialStore.get_instance()
    profile = store.get_profile(profile_name)

    if not profile:
        click.echo(f"Error: Profile '{profile_name}' not found.", err=True)
        sys.exit(1)

    if not profile.exists():
        click.echo(f"Error: Credentials file not found at {profile.resolved_path}", err=True)
        sys.exit(1)

    click.echo(f"Refreshing OAuth token for '{profile_name}'...")

    try:
        refreshed = OAuthTokenRefresher.refresh_if_needed(
            str(profile.resolved_path),
            profile.provider,
            force=force
        )

        if refreshed:
            click.echo("Token refreshed successfully!")
        else:
            if force:
                click.echo("Token refresh was not needed or failed.")
            else:
                click.echo("Token is still valid, no refresh needed.")
                click.echo("Use --force to refresh anyway.")

    except Exception as e:
        click.echo(f"Error refreshing token: {e}", err=True)
        sys.exit(1)


def _is_command_available(cmd: str) -> bool:
    """Check if a command is available in PATH."""
    import shutil
    return shutil.which(cmd) is not None
