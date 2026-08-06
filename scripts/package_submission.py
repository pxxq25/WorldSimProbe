#!/usr/bin/env python3
from worldsimprobe.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["package-submission", *__import__("sys").argv[1:]]))
