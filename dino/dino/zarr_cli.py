from __future__ import annotations

import sys
from importlib import import_module

import click


COMMANDS = {
    "periodic": ("dino.periodic", "Fit and apply shared periodic intensity correction."),
    "prepare": ("dino.zarr_prepare", "Index a masked-label Zarr for direct Cell-DINO use."),
    "barrage": ("dino.zarr_barrage", "Build a labeling barrage from direct Zarr sources."),
    "train": ("dino.zarr_train", "Train Cell-DINO from a direct-Zarr barrage."),
    "infer": ("dino.zarr_infer", "Classify cells from direct Zarr sources."),
}


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Build and run Cell-DINO directly from fused and label Zarrs."""


def _command(name: str, module_name: str, description: str) -> click.Command:
    @click.command(
        name=name,
        help=description,
        add_help_option=False,
        context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    )
    @click.argument("args", nargs=-1, type=click.UNPROCESSED)
    def command(args: tuple[str, ...]) -> None:
        sys.argv = [f"dino-zarr {name}", *args]
        import_module(module_name).main()

    return command


for command_name, (module_name, description) in COMMANDS.items():
    main.add_command(_command(command_name, module_name, description))
