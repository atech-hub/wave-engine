"""Clean Gutenberg texts: strip headers, footers, BOM, illustrations."""
import os
import re
import glob

def clean_gutenberg(text):
    # Strip BOM
    text = text.lstrip('\ufeff')

    # Find start marker
    start_markers = ['*** START OF THE PROJECT GUTENBERG', '*** START OF THIS PROJECT GUTENBERG']
    for marker in start_markers:
        idx = text.find(marker)
        if idx >= 0:
            # Skip past the marker line
            newline = text.find('\n', idx)
            if newline >= 0:
                text = text[newline + 1:]
            break
    else:
        # No standard marker — try "The Project Gutenberg eBook" header
        # Skip past the first blank line after the header block
        lines = text.split('\n')
        start = 0
        for i, line in enumerate(lines):
            if i > 5 and line.strip() == '' and i < 50:
                start = i + 1
                break
        text = '\n'.join(lines[start:])

    # Find end marker
    end_markers = ['*** END OF THE PROJECT GUTENBERG', '*** END OF THIS PROJECT GUTENBERG',
                   'End of the Project Gutenberg', 'End of Project Gutenberg']
    for marker in end_markers:
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx]
            break

    # Remove illustration captions
    text = re.sub(r'\[Illustration[^\]]*\]', '', text)

    # Remove excessive blank lines (more than 2 in a row)
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    # Strip trailing whitespace
    text = text.strip()

    return text

input_dir = os.path.dirname(os.path.abspath(__file__))
for filepath in sorted(glob.glob(os.path.join(input_dir, '*.txt'))):
    basename = os.path.basename(filepath)
    if '_clean.txt' in basename or basename == 'clean.py':
        continue

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()

    cleaned = clean_gutenberg(raw)

    out_name = basename.replace('.txt', '_clean.txt')
    out_path = os.path.join(input_dir, out_name)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)

    reduction = (1 - len(cleaned) / len(raw)) * 100
    print(f"  {basename:40s} {len(raw):>10,} -> {len(cleaned):>10,}  ({reduction:.1f}% removed)")
