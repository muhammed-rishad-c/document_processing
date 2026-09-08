"""
scrape_liquidlab_content.py

Standalone content scraper for the LiquidLab RAG chatbot (Phase 1).
NOT part of the FastAPI app — run this manually whenever the site content
needs refreshing, then feed the output .txt through the existing
/documents/upload endpoint.

Usage:
    pip install requests beautifulsoup4 lxml
    python scrape_liquidlab_content.py
    python scrape_liquidlab_content.py --output my_output.txt

Refresh workflow (when the site changes):
    1. Re-run this script -> new liquidlab_content.txt
    2. Delete the old LiquidLab document via the existing delete_document
       endpoint (cleans up Postgres + Qdrant)
    3. Upload the new .txt via the existing /documents/upload endpoint
    4. Re-run validation questions against the chatbot
"""

import argparse
import os
import time
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://trail.liquidlab.in"

# Fixed, hardcoded page list -- Phase 1 scope. Careers and Blog are
# intentionally excluded (see plan.md). Do not add a crawler/sitemap here;
# the whole point is a known, deliberate set of pages.
PAGES = [
    ("Company Overview", "/"),
    ("About Us", "/about-us"),
    ("Custom Software Development", "/services/custom-software"),
    ("Web Application Development", "/services/web-application"),
    ("Mobile App Development", "/services/mobile-application"),
    ("IoT / Embedded Software", "/services/iot-embedded"),
    ("AI / ML Solutions", "/services/ai-ml"),
    ("Cloud Solutions / SaaS", "/services/cloud-saas"),
    ("ERP / CRM / HRMS Solutions", "/services/erp-crm-hrms"),
    ("DevOps & Deployment", "/services/devops"),
    ("Workshops & Training", "/services/workshops-training"),
    ("Camera Parking Solutions", "/solutions/camera-parking"),
    ("Ultrasonic Parking Solutions", "/solutions/ultrasonic-parking"),
    ("Ticketless ANPR Parking Solutions", "/solutions/anpr-parking"),
    ("Attendance Management Solutions", "/solutions/attendance-management"),
    ("Trolley Inventory Solutions", "/solutions/trolley-inventory"),
    ("Telecom Quality of Service", "/solutions/telecom-qos"),
    ("Contact", "/contact"),
]

# Tags that are pure layout/boilerplate and never contain page-specific
# content worth keeping -- stripped before any text is pulled out.
STRUCTURAL_TAGS_TO_DROP = [
    "header", "footer", "nav", "script", "style", "noscript", "svg", "iframe",
]

# Exact-match (case-insensitive) short lines that are nav labels, CTA button
# text, or other boilerplate that survives structural stripping because it
# isn't always inside <header>/<footer>/<nav> tags in the rendered markup.
NOISE_PHRASES = {
    "home", "about us", "services", "solutions", "careers", "blog", "contact",
    "get in touch", "explore", "learn more", "read full story",
    "get started now", "book a demo", "schedule consultation", "join",
    "explore roles", "privacy", "terms", "support", "liquidlab",
    "hover to pause", "our clients", "our services", "our process",
}

MIN_LINE_LENGTH = 2  # drop empty/near-empty stray lines, keep short stats like "2015"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; LiquidLabContentBot/1.0; "
        "+internal knowledge-base refresh script)"
    )
}


@dataclass
class ScrapedPage:
    title: str
    url: str
    lines: list


def fetch_page(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=REQUEST_HEADERS, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def extract_lines(soup: BeautifulSoup) -> list:
    """Structural extraction: strip layout tags, pull remaining text as lines."""
    main = soup.find("main") or soup.body
    if main is None:
        return []
    for tag in main.find_all(STRUCTURAL_TAGS_TO_DROP):
        tag.decompose()
    text = main.get_text(separator="\n")
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if len(line) >= MIN_LINE_LENGTH]
    return lines


def dedupe_consecutive_blocks(lines: list) -> list:
    """
    Collapse repeated consecutive blocks of any size (handles the
    logo/testimonial/partner carousels that render the same content
    2-4x in a row for a scrolling-marquee UI effect).
    Generic block-repeat detection, not hardcoded to specific content.
    """
    result = []
    i = 0
    n = len(lines)
    while i < n:
        matched = False
        max_block = min(60, n - i)
        for block_size in range(max_block, 0, -1):
            if i + 2 * block_size > n:
                continue
            if lines[i:i + block_size] == lines[i + block_size:i + 2 * block_size]:
                result.extend(lines[i:i + block_size])
                j = i + block_size
                while j + block_size <= n and lines[j:j + block_size] == lines[i:i + block_size]:
                    j += block_size
                i = j
                matched = True
                break
        if not matched:
            result.append(lines[i])
            i += 1
    return result


def filter_noise(lines: list) -> list:
    return [line for line in lines if line.strip().lower() not in NOISE_PHRASES]


def scrape_page(title: str, path: str) -> ScrapedPage:
    url = BASE_URL + path
    soup = fetch_page(url)
    lines = extract_lines(soup)
    lines = dedupe_consecutive_blocks(lines)
    lines = filter_noise(lines)
    return ScrapedPage(title=title, url=url, lines=lines)


def build_combined_text(pages: list) -> str:
    sections = []
    for page in pages:
        header = f"## {page.title} \u2014 {page.url}"
        body = "\n".join(page.lines)
        sections.append(f"{header}\n\n{body}\n")
    return "\n\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description="Scrape LiquidLab site content for the RAG chatbot.")
    default_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "liquidlab_content.txt")
    parser.add_argument("--output", default=default_output, help="Output .txt path")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between page requests")
    args = parser.parse_args()

    scraped_pages = []
    for title, path in PAGES:
        print(f"Scraping: {title} ({path})")
        try:
            page = scrape_page(title, path)
            if not page.lines:
                print(f"  WARNING: no content extracted for {path} -- check manually")
            scraped_pages.append(page)
        except requests.RequestException as e:
            print(f"  FAILED: {path} -- {e}")
        time.sleep(args.delay)

    combined = build_combined_text(scraped_pages)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(combined)

    print(f"\nDone. {len(scraped_pages)}/{len(PAGES)} pages scraped.")
    print(f"Output written to: {args.output}")
    print("Next steps: manually read through the file once, then upload it "
          "via the existing /documents/upload endpoint.")


if __name__ == "__main__":
    main()
