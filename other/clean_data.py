from langdetect import detect
import string
import re

with open("data/titles.txt", "r") as f:
    lines = [line for line in f.readlines()]

ALL_LETTERS = string.ascii_letters + " .,;'-|()" + "0123456789"
ALL_LETTERS = [l for l in ALL_LETTERS]

cleaned_data = []

for line in lines:
    if detect(line) == "en":
        cleaned = ''.join(e for e in line if e in ALL_LETTERS)
        cleaned = re.sub(", ", " , ", cleaned)
        cleaned = re.sub(r"\(", r" ( ", cleaned) # TODO: Could combine this and the next one and replace with $1
        cleaned = re.sub(r"\)", r" ) ", cleaned)
        cleaned_data.append(cleaned)

with open("data/titles_cleaned.txt", "w") as f:
    for title in cleaned_data:
        f.write(f"{title}\n")

print('done')

# TODO: Could normalise better (abbreviations etc.)
# Split off punctuation -> e.g. "Hacking," currently appears in data