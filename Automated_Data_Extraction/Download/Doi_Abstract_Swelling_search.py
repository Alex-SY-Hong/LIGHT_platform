#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python Doi_Abstract_Swelling_search.py

import requests
import pandas as pd
import time
import re
import os


# ==== Clean HTML Tags ====
def clean_html(raw_html: str) -> str:
    cleanr = re.compile("<.*?>")
    return re.sub(cleanr, "", raw_html).strip()


# ==== Construct Query String ====
def build_query(words):
    """
    words: List, can contain strings or lists of strings (representing OR clauses)
    Example:
        ["PLA", ["swelling", "swelling ratio"]]
    Will generate:
        'PLA+(swelling+OR+swelling ratio)'
    """
    query_parts = []
    for word in words:
        if isinstance(word, list):  # OR clause
            query_parts.append("(" + "+OR+".join(word) + ")")
        else:
            query_parts.append(word)
    return "+".join(query_parts)


# ==== Main Crawler Logic ====
def fetch_dois_with_logic(
    and_words,
    or_sets=None,
    not_words=None,
    total=3000,
    per_page=500,
    from_pub_date="2010-01-01",
):
    """
    and_words: Keywords that must appear together (list).
    or_sets:   List of lists for OR combinations; each sublist is a set of OR keywords.
               If None, only and_words are used.
    not_words: Used to filter out records containing these words (in title + abstract).
    total:     Target maximum number of records to fetch.
    per_page:  Number of items per page for Crossref requests.
    from_pub_date: Start publication date filter (YYYY-MM-DD).
    """
    results = []
    or_sets = or_sets or [[]]

    headers = {
        "User-Agent": "MyResearchBot/1.0 (mailto:your_email)"
    }

    for or_words in or_sets:
        all_words = and_words + or_words
        query_string = build_query(all_words)
        print(f"\n🔍 Query Keyword Combination: {all_words}")

        seen_dois = set()
        empty_count = 0
        current_total = 0

        for offset in range(0, total, per_page):
            if current_total >= total:
                print("📈 Reached maximum fetch limit, terminating early.")
                break

            print(f"→ Fetching records: {offset + 1} to {offset + per_page}...")

            url = (
                "https://api.crossref.org/works"
                f"?query.bibliographic={query_string}"
                f"&rows={per_page}"
                f"&offset={offset}"
                f"&filter=type:journal-article,from-pub-date:{from_pub_date}"
            )
            data = None

            # Simple retry mechanism
            for attempt in range(3):
                try:
                    response = requests.get(url, headers=headers, timeout=20)
                    if response.status_code != 200:
                        print(f"Request failed (Status Code {response.status_code}), skipping page...")
                        break
                    data = response.json()
                    break
                except requests.exceptions.RequestException as e:
                    print(f"⚠️ Request exception: {e}, retrying {attempt + 1}/3")
                    time.sleep(2)

            if data is None:
                print("❌ Multiple retries failed, skipping this batch")
                empty_count += 1
                if empty_count >= 3:
                    print("🛑 3 consecutive failures, terminating this combination.")
                    break
                continue

            items = data.get("message", {}).get("items", [])
            if not items:
                print("📭 Current page is empty, recording empty page...")
                empty_count += 1
                if empty_count >= 3:
                    print("🛑 3 consecutive empty pages, terminating this combination.")
                    break
                continue
            else:
                empty_count = 0

            new_entries = 0
            for item in items:
                title = item.get("title", [""])[0]
                doi = item.get("DOI", "")
                if not doi or doi in seen_dois:
                    continue
                seen_dois.add(doi)

                abstract_raw = item.get("abstract", "")
                abstract = clean_html(abstract_raw) if abstract_raw else ""

                # NOT Filtering (Optional)
                if not_words:
                    lower_text = (title + " " + abstract).lower()
                    if any(nw.lower() in lower_text for nw in not_words):
                        continue

                results.append(
                    {
                        "Title": title,
                        "DOI": doi,
                        "Abstract": abstract,
                        "QueryWords": " ".join(
                            " / ".join(w) if isinstance(w, list) else w
                            for w in all_words
                        ),
                    }
                )
                new_entries += 1

            if new_entries == 0:
                print("🛑 All items on this page are duplicates, no new data, terminating this combination.")
                break

            current_total += new_entries
            time.sleep(2)

    return results


def main():
    # ==== Material Keywords ====
    names = [
        "polymer", "copolymer", "blend", "biopolymer",
        
        "PLA", "polylactic acid", "PCL", "polycaprolactone",
        "PET", "polyethylene terephthalate", "polymethyl methacrylate",
        "polyurethane", "polyamide", "nylon", "polyvinyl chloride",
        "PVA", "polyvinyl alcohol", "polyacrylonitrile", "polyvinylpyrrolidone",
        "PDMS", "polydimethylsiloxane", "polycarbonate", "polybutylene succinate",
        "PGA", "poly(γ-glutamic acid)",  "polyethylene", "polypropylene", "polyester",
        "GelMA", "gelatin methacrylate", "Gelatin", "Collagen", "Chitosan",
        "Sodium alginate", "alginate", "Cellulose", "Hyaluronic acid", 
        "silk fibroin", "polyethylene glycol", "polydopamine", "Polyacrylamide"
    ]
    # ==== Absorption-related Keyword Groups ====
    keyword_groups = [
        ["swelling ratio"],
        # If you want to extend later, e.g.:
        # ["swelling ratio", "water uptake"],
        # ["swelling ratio", "water absorption"],
    ]

    # ==== Save Path ====
    save_dir = r"your_SwellingRatio_DOI"
    os.makedirs(save_dir, exist_ok=True)

    global_seen_dois = set()
    file_counter = 1

    for name in names:
        for group in keyword_groups:
            and_group = [name] + group
            batch_result = fetch_dois_with_logic(
                and_words=and_group,
                total=3000,
                per_page=500,
            )

            new_entries = []
            for item in batch_result:
                doi = item["DOI"]
                if doi not in global_seen_dois:
                    global_seen_dois.add(doi)
                    new_entries.append(item)

            if new_entries:
                df = pd.DataFrame(new_entries)
                mat_str = name.replace(" ", "_")
                kw_str = "_".join(group)
                filename = os.path.join(
                    save_dir, f"{file_counter:04d}_{mat_str}_{kw_str}.csv"
                )
                df.to_csv(filename, index=False, encoding="utf-8-sig")
                print(f"💾 File saved: {filename}, Total {len(df)} items")
                file_counter += 1
            else:
                print(f"⚠️ Skipped: {name} + {group} (No new entries)")


if __name__ == "__main__":
    main()