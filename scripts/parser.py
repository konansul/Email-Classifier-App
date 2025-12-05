import os
import pandas as pd

def parse_raw_email_csv(input_path, output_path):

    if not os.path.exists(input_path):
        raise FileNotFoundError(f'File not found: {input_path}')

    with open(input_path, 'r', encoding = 'utf-8', errors = 'replace') as f:
        raw_lines = f.readlines()

    records = []
    in_record = False
    current_category = None
    email_parts = []

    for line in raw_lines[1:]:

        if not in_record:
            if not line.strip():
                continue

            if line.lstrip().startswith('"'):
                in_record = True
                email_parts = []

                l = line.strip()
                if l.startswith('"'):
                    l = l[1:]

                idx = l.find(',')
                if idx == -1:
                    in_record = False
                    current_category = None
                    email_parts = []
                    continue

                current_category = l[:idx].strip()
                rest = l[idx + 1:]
                email_parts.append(rest)

        else:
            email_parts.append(line)

            if line.strip().endswith('"""'):
                full_text = "".join(email_parts)

                full_text_clean = (
                    full_text.replace('"""', '')
                             .replace('""', '"')
                             .replace('"', '')
                             .strip()
                )

                records.append((current_category, full_text_clean))

                in_record = False
                current_category = None
                email_parts = []

    data = pd.DataFrame(records, columns = ['category', 'email_text'])
    data.to_csv(output_path, index = False)

    print(f"Parsed emails: {len(data)}")
    print(f"Saved to: {output_path}")

    return data

parse_raw_email_csv('../data/cat_emails_v2(in).csv', '../data/clean_email.csv')