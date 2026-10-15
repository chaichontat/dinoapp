from __future__ import annotations

import os
import sys
from importlib import import_module
from importlib.util import find_spec

import click


COMMANDS = {
    "periodic": ("dino.periodic", "main", "Fit and apply shared periodic intensity correction."),
    "barrage": ("dino.barrage", "main", "Sample cells and render a labeling barrage."),
    "clean-labels": ("dino.clean_labels", "main", "Normalize and deduplicate labels."),
    "infer": ("dino.infer", "main", "Classify cells from one dataset or ROI."),
    "label": ("dino.cli", "launch_reviewer", "Launch the standalone Streamlit reviewer."),
    "mnist": ("dino.mnist", "main", "Run the DINOv3 MNIST validation experiment."),
    "model-sweep": ("dino.model_sweep", "main", "Compare classical classifiers."),
    "rf-search": ("dino.rf_search", "main", "Tune the random-forest classifier."),
    "rf-train": ("dino.rf_train", "main", "Fit the random-forest model bundle."),
    "roc": ("dino.roc", "main", "Render ROC diagnostics."),
    "run": ("dino.runner", "main", "Run inference across datasets."),
    "train": ("dino.train", "main", "Train and evaluate Cell-DINO classifiers."),
}


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Build, label, train, and run DINO cell classifiers."""


def launch_reviewer() -> None:
    """Run the installed reviewer in the current Python environment."""
    if find_spec("dinoapp") is None:
        raise click.ClickException(
            "Install dinoapp in this Python environment before running dino label: "
            f"{sys.executable} -m pip install dinoapp"
        )
    os.execv(sys.executable, [sys.executable, "-m", "dinoapp", *sys.argv[1:]])


def _command(
    name: str,
    module_name: str,
    function_name: str,
    description: str,
) -> click.Command:
    @click.command(
        name=name,
        help=description,
        add_help_option=False,
        context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    )
    @click.argument("args", nargs=-1, type=click.UNPROCESSED)
    def command(args: tuple[str, ...]) -> None:
        sys.argv = [f"dino {name}", *args]
        getattr(import_module(module_name), function_name)()

    return command


for command_name, (module_name, function_name, description) in COMMANDS.items():
    main.add_command(_command(command_name, module_name, function_name, description))
