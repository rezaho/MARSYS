"""
MARSYS CLI - Command Line Interface for the Multi-Agent Reasoning System.

Provides commands for managing OAuth credentials, profiles, and other
framework configuration.

Usage:
    marsys --help
    marsys oauth --help
    marsys oauth list
    marsys oauth add work-claude --provider anthropic-oauth
"""

import click

from .oauth import oauth


@click.group()
@click.version_option(package_name="marsys")
def main():
    """MARSYS - Multi-Agent Reasoning System CLI.

    A powerful framework for orchestrating collaborative AI agents with
    sophisticated reasoning, planning, and autonomous capabilities.
    """
    pass


# Register command groups
main.add_command(oauth)


if __name__ == "__main__":
    main()
