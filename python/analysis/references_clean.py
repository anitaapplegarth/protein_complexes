    #!/usr/bin/env python3
"""
Strip a Zotero-exported .bib down to the fields OUP author-date style needs.

Set SRC and DST below, then run.

Keeps only the fields listed in KEEP per entry type, repairs prefixed page
ranges (D646--650 -> D646--D650), and reports entries that are missing
fields the reference list will need.

No dependencies; brace-aware parser, so nested braces in titles survive.
"""

import re
from collections import OrderedDict

# ---------------------------------------------------------------------
# EDIT THESE TWO LINES, then press Run.
# Use the full path if the .bib is not in the same folder as this script,
# e.g. SRC = "/Users/anita/Downloads/references.bib"
SRC = "/Users/anitaapplegarth/Desktop/references.bib"
DST = "/Users/anitaapplegarth/Desktop/references_clean.bib"
# ---------------------------------------------------------------------

# Fields to keep, by entry type. Anything not listed is dropped.
KEEP = {
    "article":       ["author", "title", "journal", "year", "volume", "pages", "doi"],
    "inproceedings": ["author", "title", "booktitle", "year", "pages", "doi"],
    "incollection":  ["author", "editor", "title", "booktitle", "publisher",
                      "year", "pages", "doi"],
    "book":          ["author", "editor", "title", "publisher", "year", "edition"],
    "phdthesis":     ["author", "title", "school", "year", "url"],
    # Websites and preprints keep url + urldate: the style formats them that way.
    "misc":          ["author", "title", "publisher", "year", "doi", "url",
                      "urldate", "note"],
}
DEFAULT_KEEP = ["author", "editor", "title", "year", "doi", "url", "urldate"]

# Entries warned about if these are absent.
REQUIRED = {
    "article":       ["author", "title", "journal", "year", "volume", "pages"],
    "inproceedings": ["author", "title", "booktitle", "year"],
    "misc":          ["author", "title", "year"],
}


def split_entries(text):
    """Yield (entry_type, citekey, body) for each @type{...} block."""
    for m in re.finditer(r"@(\w+)\s*\{", text):
        etype = m.group(1).lower()
        if etype in ("comment", "string", "preamble"):
            continue
        i = m.end()
        depth = 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        block = text[m.end():i - 1]
        key, _, body = block.partition(",")
        yield etype, key.strip(), body


def parse_fields(body):
    """Return OrderedDict of field -> raw value string."""
    fields = OrderedDict()
    i = 0
    n = len(body)
    while i < n:
        m = re.compile(r"\s*([A-Za-z][\w-]*)\s*=\s*").match(body, i)
        if not m:
            break
        name = m.group(1).lower()
        i = m.end()
        if i < n and body[i] == "{":
            depth, start = 1, i
            i += 1
            while i < n and depth:
                if body[i] == "{":
                    depth += 1
                elif body[i] == "}":
                    depth -= 1
                i += 1
            value = body[start:i]
        elif i < n and body[i] == '"':
            start = i
            i += 1
            while i < n and body[i] != '"':
                i += 1
            i += 1
            value = body[start:i]
        else:                                   # bare value, e.g. month = may
            start = i
            while i < n and body[i] not in ",\n":
                i += 1
            value = body[start:i].strip()
        fields[name] = value
        while i < n and body[i] in ", \n\t":
            i += 1
    return fields


def fix_pages(value):
    """D646--650 -> D646--D650. Leaves everything else alone."""
    inner = value.strip()
    stripped = inner[1:-1] if inner.startswith("{") else inner
    m = re.fullmatch(r"([A-Za-z]+)(\d+)\s*--\s*(\d+)", stripped.strip())
    if m:
        fixed = f"{m.group(1)}{m.group(2)}--{m.group(1)}{m.group(3)}"
        return "{" + fixed + "}"
    return value


def main(src, dst):
    text = open(src, encoding="utf-8").read()
    out, warnings, count = [], [], 0

    for etype, key, body in split_entries(text):
        fields = parse_fields(body)
        keep = KEEP.get(etype, DEFAULT_KEEP)
        kept = OrderedDict()
        for name in keep:
            if name in fields:
                value = fields[name]
                if name == "pages":
                    value = fix_pages(value)
                kept[name] = value

        for name in REQUIRED.get(etype, []):
            if name not in kept:
                warnings.append(f"  {key}: missing {name}")

        lines = [f"@{etype}{{{key},"]
        lines += [f"\t{n} = {v}," for n, v in kept.items()]
        lines.append("}")
        out.append("\n".join(lines))
        count += 1

    open(dst, "w", encoding="utf-8").write("\n\n".join(out) + "\n")
    print(f"Wrote {count} entries to {dst}")
    if warnings:
        print(f"\n{len(warnings)} entries need attention:")
        print("\n".join(warnings))


if __name__ == "__main__":
    main(SRC, DST)