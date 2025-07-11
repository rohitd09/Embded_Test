from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from bs4.element import Comment

# Filter visible text
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

# Extract all visible text from a rendered page
def get_text_from_js_page(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000)
        page.wait_for_load_state("networkidle")

        html = page.content()
        browser.close()

    soup = BeautifulSoup(html, 'html.parser')
    texts = soup.findAll(string=True)
    visible_texts = filter(tag_visible, texts)
    return "\n".join(t.strip() for t in visible_texts if t.strip())

# Run
url = "https://www.apple.com/legal/privacy/en-ww/"
visible_text = get_text_from_js_page(url)

# Save to file
with open("cache.txt", "w", encoding="utf-8") as f:
    f.write(visible_text)