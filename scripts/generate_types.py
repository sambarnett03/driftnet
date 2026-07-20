from pathlib import Path


def generate_experiment_types(
        target_file:
        str = "/home/users/sbarnett/documents/driftnet/src/driftnet/generated_types.py"
    ):
    base_dir = Path("/gws/ssde/j25b/oxford_es/sbarnett/driftnet/experiments")

    if not base_dir.exists():
        print("'experiments/' directory not found. Skipping type generation.")
        return

    experiment_paths = []

    # Walk through experiments and their trials
    for exp_dir in base_dir.iterdir():
        if exp_dir.is_dir() and not exp_dir.name.startswith("."):
            for trial_dir in exp_dir.iterdir():
                if trial_dir.is_dir() and not trial_dir.name.startswith("."):
                    # Use .as_posix() to ensure forward slashes ("exp1/trial1") even on Windows
                    combined_path = (Path(exp_dir.name) / trial_dir.name).as_posix()
                    experiment_paths.append(combined_path)

    # Sort for clean, predictable file output
    experiment_paths.sort()

    # Write the python file
    with open(target_file, "w") as f:
        f.write("# AUTO-GENERATED FILE. DO NOT EDIT DIRECTLY.\n")
        f.write("# Run `python generate_types.py` to update these types.\n\n")
        f.write("from typing import Literal\n\n")

        if experiment_paths:
            path_literals = ", ".join(f'"{p}"' for p in experiment_paths)
            f.write(f"ExperimentPathType = Literal[{path_literals}]\n")
        else:
            f.write("ExperimentPathType = str\n")

        f.write("# type: ignore")

    print(f"Successfully updated combined types in {target_file}!")


if __name__ == "__main__":
    generate_experiment_types()
