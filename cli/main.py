"""
Main CLI entry point for SearchAll application.
"""

import click
from cli.commands import index, search, ask


@click.group()
@click.version_option(version="0.1.0")
@click.pass_context
def cli(ctx):
    """
    Hyperthymesia - Index and search your local files.

    A powerful tool to index your data and search through it quickly.
    """
    # Ensure that ctx.obj exists and is a dict
    ctx.ensure_object(dict)


# Add command groups
cli.add_command(index)
cli.add_command(search)
cli.add_command(ask)

if __name__ == "__main__":
    cli()
