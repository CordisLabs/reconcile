"""Download a public-domain corpus for the scaling experiment.

Three Project Gutenberg books, concatenated. Both are public domain; the texts are not
committed — this script fetches them so the repository carries no redistribution.
"""

import os
import time
import urllib.request

BOOKS = {
    "11": "Alice's Adventures in Wonderland",
    "2701": "Moby Dick",
    "2600": "War and Peace",
    "1661": "The Adventures of Sherlock Holmes",
    "1342": "Pride and Prejudice",
    "98": "A Tale of Two Cities",
    "1400": "Great Expectations",
    "74": "The Adventures of Tom Sawyer",
    "345": "Dracula",
    "5200": "Metamorphosis",
    "84": "Frankenstein",
    "45": "Anne of Green Gables",
    "76": "Adventures of Huckleberry Finn",
    "2554": "Crime and Punishment",
    "28054": "The Brothers Karamazov",
    "1399": "Anna Karenina",
    "46": "A Christmas Carol",
    "4300": "Ulysses",
    "1232": "The Prince",
    "16328": "Beowulf",
    "17489": "The Yellow Wallpaper",
    "1952": "The Yellow Wallpaper",
    "219": "Heart of Darkness",
    "160": "The Awakening",
    "1260": "Jane Eyre",
    "768": "Wuthering Heights",
    "145": "Middlemarch",
    "158": "Emma",
    "161": "Sense and Sensibility",
    "105": "Persuasion",
    "1023": "Bleak House",
    "766": "David Copperfield",
    "564": "The Mystery of Edwin Drood",
    "245": "Life on the Mississippi",
    "3176": "The Innocents Abroad",
    "3437": "The Idiot",
    "600": "Notes from Underground",
}
BASE = "https://www.gutenberg.org/files/{0}/{0}-0.txt"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.txt")


def strip_gutenberg(text):
    start = text.find("*** START OF")
    end = text.find("*** END OF")
    if start != -1:
        text = text[text.find("\n", start) + 1:]
    if end != -1:
        text = text[:end]
    return text


def main():
    parts = []
    for book, title in BOOKS.items():
        url = BASE.format(book)
        raw = None
        for attempt in range(3):
            try:
                print(f"fetching {title} (try {attempt + 1}) …", flush=True)
                with urllib.request.urlopen(url, timeout=120) as fh:
                    raw = fh.read().decode("utf-8", errors="ignore")
                break
            except Exception as exc:                     # noqa: BLE001
                print(f"  failed: {exc}", flush=True)
                time.sleep(3)
        if raw is None:
            print(f"  skipping {title}", flush=True)
            continue
        parts.append(strip_gutenberg(raw))
    text = "\n".join(parts)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {OUT}: {len(text):,} characters from {len(BOOKS)} books")


if __name__ == "__main__":
    main()
