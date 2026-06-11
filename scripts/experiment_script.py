import argparse


def main():
    parser = argparse.ArgumentParser(description="Run Driftnet")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to the config file"
    )
    # args = parser.parse_args()

    # Load the configuration file
    # with open(args.config) as f:
    #     config = yaml.safe_load(f)

    # Pass the config dictionary to your core logic
    # run_model(config)


if __name__ == "__main__":
    main()
