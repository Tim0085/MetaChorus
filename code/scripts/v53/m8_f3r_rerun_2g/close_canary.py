from __future__ import annotations

import argparse

from metachorus_v53.m8_f3r_rerun_2g.stages import close_canary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args()
    value = close_canary(args.result_root)
    print({"status": value["status"], "identity": value["identity"]})


if __name__ == "__main__":
    main()
