# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | yes |
| < 0.1 | no |

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue.
Use the repository's private vulnerability reporting on GitHub
("Security" → "Report a vulnerability"), or open an issue that asks
privately for a contact method if that is not available.

Include: what you did, what you expected, what happened, and the version
or commit.

You can expect an acknowledgement within a week and an assessment within
two.

## What this project does and does not do with credentials

This package downloads **public market data only**:

- Binance's public REST klines endpoint. No API key, no account, no
  signature.
- Yahoo Finance via `yfinance`, which is a public quote history.
- Dukascopy's public `.bi5` tick archive.
- Any CSV you drop into the directory yourself.

It never sends your trades, your account, or your credentials anywhere. It
does not read any credential file. If you point it at a CSV, that file
stays on your machine.

That said:

- **Do not commit market data or credentials.** The `.gitignore` already
  excludes `data/`, `*.csv`, `*.parquet`, `*.bi5` and `*.zip`. Keep it that
  way, and check what you are about to stage.
- **The rulesets in `src/prop_lab/firm/rules.py` are public information**
  (a firm's published evaluation terms). They are not secrets, but they
  do go stale, which is a correctness problem rather than a security one.
  Each one carries a `rules_checked` date for that reason.

## Running untrusted strategy files

`proplab --strategy path/to/file.py:Class` executes the Python file you
point it at, with your privileges. Only run strategy files you have read.
This is the same trust model as any other Python script you run, and it is
not a sandbox.
