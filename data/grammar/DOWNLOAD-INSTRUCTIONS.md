# Grammar Text Download — Working Mirror URLs
# Date: 2026-03-22
# Note: gutenberg.org is returning 504 errors. Use the mirror below.

## Working mirror: mirrorservice.org

Code should download these using PowerShell (on Marco's Windows PC):

```powershell
# Create directory
mkdir C:\claude\wave-engine\data\grammar -Force

# 1. Plain English — Marian Wharton (beginner)
Invoke-WebRequest -Uri "http://www.mirrorservice.org/sites/gutenberg.org/4/0/5/5/40550/40550-0.txt" -OutFile "C:\claude\wave-engine\data\grammar\01_plain_english.txt"

# 2. Practical Grammar — Thomas Wood
Invoke-WebRequest -Uri "http://www.mirrorservice.org/sites/gutenberg.org/2/2/5/7/22577/22577-0.txt" -OutFile "C:\claude\wave-engine\data\grammar\02_practical_grammar.txt"

# 3. An English Grammar — Baskervill & Sewell
Invoke-WebRequest -Uri "http://www.mirrorservice.org/sites/gutenberg.org/1/4/0/0/14006/14006-0.txt" -OutFile "C:\claude\wave-engine\data\grammar\03_english_grammar.txt"

# 4. Advanced English Grammar — Kittredge & Farley
Invoke-WebRequest -Uri "http://www.mirrorservice.org/sites/gutenberg.org/4/5/8/1/45814/45814-0.txt" -OutFile "C:\claude\wave-engine\data\grammar\04_advanced_grammar.txt"

# 5. Word Study and English Grammar — Hamilton
Invoke-WebRequest -Uri "http://www.mirrorservice.org/sites/gutenberg.org/3/0/0/3/30036/30036-0.txt" -OutFile "C:\claude\wave-engine\data\grammar\05_word_study.txt"

# Verify downloads
Get-ChildItem C:\claude\wave-engine\data\grammar\*.txt | ForEach-Object { "$($_.Name): $($_.Length / 1KB)KB" }
```

## If mirror also fails

Marco can download manually in his browser:
- Go to: http://www.mirrorservice.org/sites/gutenberg.org/4/0/5/5/40550/
- Click 40550-0.txt
- Save to C:\claude\wave-engine\data\grammar\01_plain_english.txt
- Repeat for each book

## After download: Clean and concatenate

```powershell
# Concatenate all files in order
Get-Content C:\claude\wave-engine\data\grammar\01_*.txt,
            C:\claude\wave-engine\data\grammar\02_*.txt,
            C:\claude\wave-engine\data\grammar\03_*.txt,
            C:\claude\wave-engine\data\grammar\04_*.txt,
            C:\claude\wave-engine\data\grammar\05_*.txt |
    Set-Content C:\claude\wave-engine\data\grammar\grammar_corpus.txt

# Check total size
(Get-Item C:\claude\wave-engine\data\grammar\grammar_corpus.txt).Length / 1MB
```

## Gutenberg text file URL pattern
The pattern is: mirrorservice.org/sites/gutenberg.org/{d1}/{d2}/{d3}/{d4}/{id}/{id}-0.txt
Where d1/d2/d3/d4 are the digits of the ID minus the last digit.
Example: ID 40550 → 4/0/5/5/40550/40550-0.txt
