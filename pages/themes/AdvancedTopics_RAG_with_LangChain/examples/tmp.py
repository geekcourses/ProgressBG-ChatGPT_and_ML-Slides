from langchain_text_splitters import MarkdownHeaderTextSplitter

# Convert your text to Markdown with headers
text = """##
If the application fails to launch, first ensure your operating system is up to date.

## For Windows 11 users, verify that Windows Defender is not blocking the executable.

## Clearing the application cache in the AppData/Local folder often resolves sync issues."""


# Define headers to split on
headers_to_split_on = [
    ("##", "Header2"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

paragraphs = markdown_splitter.split_text(text)

print(f"Number of paragraphs: {len(paragraphs)}")
for i, para in enumerate(paragraphs, 1):
    print(f"Paragraph {i}: {para.page_content.strip()}")
    print("-" * 50)
