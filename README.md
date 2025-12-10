# NGO-mission-website-analysis
import re
import json
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
import textstat
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import yake
from slugify import slugify

STOPWORDS = set(stopwords.words('english'))
USER_AGENT = "Mozilla/5.0 (NGO-Mission-Analyzer/1.0)"

def fetch_url(url, timeout=12):
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text, r.url

def get_domain(url):
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"

def parse_html(html):
    return BeautifulSoup(html, "html.parser")

def extract_meta(soup):
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    m = soup.find("meta", attrs={"name": "description"})
    meta_desc = m["content"].strip() if m and m.get("content") else ""
    m2 = soup.find("meta", attrs={"name": "keywords"})
    meta_keywords = m2["content"].strip() if m2 and m2.get("content") else ""
    return {"title": title, "meta_description": meta_desc, "meta_keywords": meta_keywords}

def extract_headings(soup):
    headings = {}
    for i in range(1, 7):
        tags = [t.get_text(" ", strip=True) for t in soup.find_all(f"h{i}")]
        headings[f"h{i}"] = tags
    return headings

def get_all_text(soup):
    for s in soup(["script", "style", "noscript", "svg", "iframe"]):
        s.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r'\n\s*\n+', '\n\n', text).strip()
    return text

def extract_mission_sentences(soup, heuristics=None):
    heuristics = heuristics or ["mission", "our mission", "vision and mission", "mission statement", "what we do"]
    found = []
    for tag in soup.find_all(re.compile("^h[1-6]$")):
        txt = tag.get_text(" ", strip=True).lower()
        for h in heuristics:
            if h in txt:
                for sib in tag.find_next_siblings(limit=6):
                    if sib.name in ["p", "div", "span", "li"]:
                        t = sib.get_text(" ", strip=True)
                        if t and len(t) > 20:
                            found.append(t)
                break
    for p in soup.find_all(["p", "li"]):
        t = p.get_text(" ", strip=True)
        if len(t) > 40 and re.search(r'\bmission\b', t, re.I):
            found.append(t)
    unique = []
    for s in found:
        if s not in unique:
            unique.append(s)
    return unique

def extract_links_and_images(soup, base_url):
    links = []
    images = []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        text = a.get_text(" ", strip=True)
        links.append({"href": href, "text": text})
    for img in soup.find_all("img"):
        src = img.get("src") or ""
        alt = img.get("alt") or ""
        images.append({"src": urljoin(base_url, src), "alt": alt})
    return links, images

def extract_contacts(text):
    emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    phones = re.findall(r'(\+?\d[\d\-\s\(\)]{7,}\d)', text)
    return list(dict.fromkeys(emails)), list(dict.fromkeys(phones))

def extract_keywords(text, max_keywords=10):
    kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=max_keywords, features=None)
    keywords = kw_extractor.extract_keywords(text)
    return [kw for kw, score in keywords]

def top_n_terms(text, n=10):
    vec = TfidfVectorizer(stop_words='english', max_features=2000)
    t = vec.fit_transform([text])
    sums = t.sum(axis=0)
    terms = [(word, sums[0, idx]) for word, idx in vec.vocabulary_.items()]
    terms = sorted(terms, key=lambda x: x[1], reverse=True)[:n]
    return [t[0] for t in terms]

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(text)

def readability_scores(text):
    try:
        flesch = textstat.flesch_reading_ease(text)
        grade = textstat.text_standard(text, float_output=True)
        smog = textstat.smog_index(text)
    except:
        flesch = grade = smog = None
    counts = {
        "words": len(re.findall(r'\w+', text)),
        "sentences": len(sent_tokenize(text))
    }
    return {"flesch_reading_ease": flesch, "estimated_grade": grade, "smog_index": smog, **counts}

def seo_checks(meta, headings, links, images, base_url):
    title = meta.get("title", "")
    desc = meta.get("meta_description", "")
    issues = []
    checks = {}
    checks['title_length'] = len(title)
    checks['meta_description_length'] = len(desc)
    checks['has_h1'] = len(headings.get("h1", [])) > 0
    checks['num_internal_links'] = len([l for l in links if urlparse(l['href']).netloc == urlparse(base_url).netloc])
    checks['num_external_links'] = len(links) - checks['num_internal_links']
    if checks['title_length'] == 0:
        issues.append("Missing <title> tag")
    if checks['title_length'] > 70:
        issues.append("Title looks long (>70 chars)")
    if checks['meta_description_length'] == 0:
        issues.append("Missing meta description")
    if checks['meta_description_length'] > 160:
        issues.append("Meta description looks long (>160 chars)")
    images_missing_alt = [img['src'] for img in images if not img.get('alt')]
    checks['images_missing_alt'] = images_missing_alt[:20]
    if images_missing_alt:
        issues.append(f"{len(images_missing_alt)} images missing alt text")
    checks['issues'] = issues
    return checks

def accessibility_checks(soup):
    issues = []
    forms = soup.find_all("form")
    forms_without_labels = 0
    for f in forms:
        inputs = f.find_all(["input", "select", "textarea"])
        for inp in inputs:
            if inp.name == "input" and inp.get("type") in ["hidden", "submit", "button"]:
                continue
            has_label = False
            if inp.get("id"):
                if soup.find("label", attrs={"for": inp.get("id")}):
                    has_label = True
            if inp.find_parent("label"):
                has_label = True
            if not has_label:
                forms_without_labels += 1
    if forms_without_labels:
        issues.append(f"{forms_without_labels} form inputs appear without labels")
    return {"forms_with_missing_labels": forms_without_labels, "issues": issues}

def safe_text_snippet(s, n=300):
    return (s[:n] + '...') if len(s) > n else s

def save_report_json(report, outname):
    with open(outname, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

def make_html_summary(report, outname):
    html = []
    html.append(f"<h1>NGO Website Analysis Report: {report.get('url','')}</h1>")
    html.append(f"<h2>Title</h2><p>{report['meta']['title']}</p>")
    html.append(f"<h2>Meta Description</h2><p>{report['meta']['meta_description']}</p>")
    html.append(f"<h2>Mission-like Texts</h2>")
    for m in report['mission_texts'][:10]:
        html.append(f"<blockquote>{m}</blockquote>")
    html.append("<h2>Top Keywords</h2><ul>")
    for k in report['keywords']:
        html.append(f"<li>{k}</li>")
    html.append("</ul>")
    html.append("<h2>Sentiment (compound)</h2><p>{}</p>".format(report['sentiment'].get('compound')))
    html.append("<h2>SEO Checks</h2><pre>{}</pre>".format(json.dumps(report['seo_checks'], indent=2)))
    html.append("<h2>Contacts Found</h2><pre>{}</pre>".format(json.dumps(report.get('contacts', {}), indent=2)))
    html.append("<h2>Accessibility Issues</h2><pre>{}</pre>".format(json.dumps(report.get('accessibility', {}), indent=2)))
    html_content = "<html><body style='font-family:Arial,Helvetica,sans-serif'>" + "\n".join(html) + "</body></html>"
    with open(outname, "w", encoding="utf-8") as f:
        f.write(html_content)

def analyze_url(url, save_json=True, save_html=True, out_prefix=None):
    if not out_prefix:
        out_prefix = slugify(urlparse(url).netloc or "report")
    report = {"url": url, "errors": []}
    try:
        html, final_url = fetch_url(url)
    except Exception as e:
        report['errors'].append(str(e))
        return report
    base = get_domain(final_url)
    soup = parse_html(html)
    report['final_url'] = final_url
    report['domain'] = base
    report['meta'] = extract_meta(soup)
    report['headings'] = extract_headings(soup)
    text = get_all_text(soup)
    try:
        report['language'] = detect(text[:1000]) if text and len(text) > 50 else "unknown"
    except:
        report['language'] = "unknown"
    mission_texts = extract_mission_sentences(soup)
    if not mission_texts:
        sentences = sent_tokenize(text)
        mission_texts = [s for s in sentences if re.search(r'\bmission\b', s, re.I)]
    report['mission_texts'] = mission_texts
    links, images = extract_links_and_images(soup, base)
    report['internal_links_count'] = len([l for l in links if urlparse(l['href']).netloc == urlparse(base).netloc])
    report['external_links_count'] = len(links) - report['internal_links_count']
    report['links_sample'] = links[:50]
    report['images_sample'] = images[:50]
    emails, phones = extract_contacts(text)
    report['contacts'] = {"emails": emails, "phones": phones}
    full_text = " ".join(report['mission_texts']) if report['mission_texts'] else text[:10000]
    report['keywords'] = extract_keywords(full_text, max_keywords=15)
    report['top_terms'] = top_n_terms(full_text, n=10)
    report['sentiment'] = analyze_sentiment(full_text)
    report['readability'] = readability_scores(full_text)
    report['seo_checks'] = seo_checks(report['meta'], report['headings'], links, images, base)
    report['accessibility'] = accessibility_checks(soup)
    report['mission_clarity'] = {
        "has_mission_section": bool(report['mission_texts']),
        "mission_length_chars": [len(m) for m in report['mission_texts']]
    }
    if save_json:
        try:
            json_out = f"{out_prefix}_ngo_report.json"
            save_report_json(report, json_out)
            report['json_report'] = json_out
        except Exception as e:
            report['errors'].append(f"Failed to save JSON: {e}")
    if save_html:
        try:
            html_out = f"{out_prefix}_ngo_report.html"
            make_html_summary(report, html_out)
            report['html_report'] = html_out
        except Exception as e:
            report['errors'].append(f"Failed to save HTML: {e}")
    return report

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ngo_mission_analysis.py <url>")
        sys.exit(1)
    url = sys.argv[1]
    r = analyze_url(url)
    print("Report saved:", r.get('json_report'), r.get('html_report'))
    if r.get('errors'):
        print("Errors:", r['errors'])
    else:
        print("Title:", r['meta']['title'])
        print("Language:", r['language'])
        print("Has mission section:", r['mission_clarity']['has_mission_section'])
        print("Top keywords:", r['keywords'][:10])
        print("Contact emails found:", r['contacts']['emails'])
        print("Sentiment (compound):", r['sentiment']['compound'])
