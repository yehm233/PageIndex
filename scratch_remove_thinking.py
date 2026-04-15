import re

file_path = "pageindex/page_index.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Pattern to remove lines containing "thinking": <...>
# that appear within prompt strings.
content = re.sub(r'\s*\"thinking\".*?\n', '\n', content)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
print("Removed thinking fields.")
