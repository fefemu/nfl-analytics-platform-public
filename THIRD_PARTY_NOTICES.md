# Third-party data and trademarks

This project combines original source code and derived analytical outputs with
data obtained from third-party sources. Publication of this repository does not
transfer or expand any rights in those sources.

## nflverse

Schedule, play-by-play, player, roster, depth-chart, injury, snap-count and team
metadata are obtained through the nflverse ecosystem. The nflverse-data
repository is published under CC BY 4.0 and requires attribution. NFL data
accessed through nflverse may also be subject to rights and terms of the
underlying data owners.

- Project: https://github.com/nflverse/nflverse-data
- License: https://github.com/nflverse/nflverse-data/blob/main/LICENSE.md

Downloaded nflverse datasets are deliberately excluded from this repository.
They can be recreated locally through the ingestion pipeline.

## nfelo and nfelounits

External team-strength ratings, QB adjustments, published probabilities and
unit-level signals are sourced from the nfelo/nfelounits ecosystem. They are
used as model inputs with visible attribution. The upstream nfelo repository
does not currently present a clearly identified repository license; therefore
its source data and outputs are not redistributed as part of this repository.

- nfelo: https://github.com/greerreNFL/nfelo
- nfelounits: https://github.com/nfelodcm/nfelounits

## The Odds API

Current Moneyline, Spread and Total prices are obtained from The Odds API. Its
terms allow use inside websites, dashboards and analytical tools, but prohibit
redistributing the data as a standalone feed, bulk export or database dump.
Raw odds snapshots and public downloadable database artifacts containing odds
data must therefore not be committed or published with this repository.

- Service: https://the-odds-api.com/
- Terms: https://the-odds-api.com/terms-and-conditions.html

Odds can change quickly. Displayed prices must be treated as time-stamped
analytical inputs and independently verified before use.

## Team names, logos and trademarks

NFL team names, logos and trademarks belong to their respective owners. The
dashboard references remote team-logo URLs for identification and descriptive
presentation; no logo files are stored in this repository. This project is
independent and is not affiliated with, endorsed by or sponsored by the NFL,
ESPN, any NFL club or any bookmaker.

Remote image availability does not itself grant a reuse license. A public or
commercial launch should replace these images with expressly licensed assets
or obtain written permission if the intended use requires it.

## Python dependencies

Python packages listed in `requirements.txt` retain their respective upstream
licenses. Installing the requirements does not make those packages part of the
copyright grant for this repository.
