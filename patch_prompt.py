import re

with open('pageindex/page_index.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Add CRITICAL INSTRUCTION into all prompt templates that have "Directly return the final JSON"
text = re.sub(
    r'(Directly return the[^."]*JSON.*?)([\'"]{3})',
    r'\1\n    CRITICAL INSTRUCTION: DO NOT INCLUDE <think> TAGS OR ANY REASONING. OUTPUT JSON DIRECTLY.\2',
    text
)

with open('pageindex/page_index.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Patched.")
