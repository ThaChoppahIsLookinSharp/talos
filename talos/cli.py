import argparse

def main():
    parser = argparse.ArgumentParser(prog="talos")
    parser.add_argument("command", choices=["run", "test"])
    args = parser.parse_args()

    if args.command == "run":
        print("Running TALOS...")
    elif args.command == "test":
        print("Testing...")
