"""Monthly arXiv watch (spec §2 and §13): new papers close to this project.

    uv run python scripts/arxiv_watch.py --days 35 > watch.md

Queries the arXiv API (export.arxiv.org, 3 s between requests as its terms ask) and prints
the papers submitted in the last ``--days`` days as Markdown. The monthly workflow
(.github/workflows/literature.yml) opens an issue when something is found; the findings
that matter go to docs/literature-watch.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
import xml.etree.ElementTree as ET

import httpx

from ofs.ingest.download import ssl_context

API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
QUERIES: dict[str, str] = {
    "order flow and crypto": 'abs:"order flow" AND (abs:crypto OR abs:bitcoin OR abs:cryptocurrency)',
    "roughness and crypto": (
        '(abs:"rough volatility" OR abs:roughness OR abs:"Hurst") '
        "AND (abs:crypto OR abs:bitcoin OR abs:cryptocurrency)"
    ),
    "order flow, impact and rough volatility": (
        'abs:"order flow" AND abs:"rough" AND (abs:impact OR abs:volume)'
    ),
    "authors of the theory": 'au:"Muhle-Karbe" OR au:"Ouazzani Chahdi"',
}


def search(client: httpx.Client, query: str, since: dt.date) -> list[dict[str, str]]:
    params = {
        "search_query": query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": "50",
    }
    resp = client.get(API, params=params)
    resp.raise_for_status()
    found = []
    for entry in ET.fromstring(resp.text).iter(f"{ATOM}entry"):
        published = dt.date.fromisoformat(entry.findtext(f"{ATOM}published", "")[:10])
        if published < since:
            continue
        authors = [a.findtext(f"{ATOM}name", "") for a in entry.iter(f"{ATOM}author")]
        found.append(
            {
                "date": published.isoformat(),
                "title": " ".join(entry.findtext(f"{ATOM}title", "").split()),
                "authors": ", ".join(authors[:4]) + (" et al." if len(authors) > 4 else ""),
                "url": entry.findtext(f"{ATOM}id", "").strip(),
            }
        )
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=35)
    args = parser.parse_args()
    since = dt.datetime.now(dt.UTC).date() - dt.timedelta(days=args.days)
    lines = [f"# arXiv watch since {since}", ""]
    seen: set[str] = set()
    with httpx.Client(verify=ssl_context(), timeout=60, follow_redirects=True) as client:
        for i, (label, query) in enumerate(QUERIES.items()):
            if i:
                time.sleep(3)
            hits = [h for h in search(client, query, since) if h["url"] not in seen]
            seen.update(h["url"] for h in hits)
            lines.append(f"## {label} ({len(hits)})")
            lines += [f"* {h['date']} [{h['title']}]({h['url']}) - {h['authors']}" for h in hits]
            lines.append("")
    print("\n".join(lines) if seen else f"No new paper since {since}.")


if __name__ == "__main__":
    main()
