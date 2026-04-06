import os
import pandas as pd

# --- Paths (relative to project root) ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATA_DIR   = os.path.join(BASE_DIR, "Dataset")

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV       = os.path.join(OUTPUT_DIR, "clean_data.csv")
EXTRACTED_FOLDER = os.path.join(DATA_DIR, "extracted")
DAT_FOLDERS      = [
    os.path.join(DATA_DIR, "state_data", "by_month2011"),
    os.path.join(DATA_DIR, "state_data", "by_month2012"),
    os.path.join(DATA_DIR, "state_data", "by_month2013"),
]

# --- Station IDs ---
# Both IDs refer to the same physical station in Randolph, Alabama
# 01014000 = pre-1993 station code
# 01014007 = post-1993 station code
STATION_IDS = {"01014000", "01014007"}


def parse_file(filepath):
    """
    Parse a single DSI-3260 .dat file.
    Returns a list of dicts with keys:
        year, month, day, hour, minute, precip_mm
    """
    records = []

    with open(filepath, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")

            # Minimum length check
            if len(line) < 30:
                continue

            # Only process 15-minute precipitation records
            record_type  = line[0:3].strip()
            element_type = line[11:15].strip()
            if record_type != "15M" or element_type != "QPCP":
                continue

            # Filter to our chosen station
            station_id = line[3:11].strip()
            if station_id not in STATION_IDS:
                continue

            # Parse header fields
            units = line[15:17].strip()
            try:
                year  = int(line[17:21])
                month = int(line[21:23])
                day   = int(line[23:27])
                n_val = int(line[27:30])
            except ValueError:
                continue

            # Parse each time-value-flag triplet
            pos = 30
            for _ in range(n_val):
                if pos + 12 > len(line):
                    break

                time_str  = line[pos:pos+4]
                value_str = line[pos+4:pos+10].strip()
                flag1     = line[pos+10] if len(line) > pos + 10 else " "
                pos += 12

                # Skip daily total (time code 2500)
                if time_str == "2500":
                    continue

                # Skip missing and deleted values
                if value_str == "099999":
                    continue
                if flag1 in ("[", "]", "{", "}", "a", ","):
                    continue

                try:
                    raw_value = int(value_str)
                except ValueError:
                    continue

                # Convert to mm
                if units in ("HI", "HT"):
                    precip_mm = raw_value / 100.0 * 25.4
                else:
                    continue

                # Parse time
                try:
                    hour   = int(time_str[0:2])
                    minute = int(time_str[2:4])
                    if hour == 24:
                        hour = 0
                except ValueError:
                    continue

                records.append({
                    "year":      year,
                    "month":     month,
                    "day":       day,
                    "hour":      hour,
                    "minute":    minute,
                    "precip_mm": precip_mm
                })

    return records


# --- Main ---
all_records = []

# Scan extracted folder (files have NO extension)
if os.path.isdir(EXTRACTED_FOLDER):
    for filename in sorted(os.listdir(EXTRACTED_FOLDER)):
        filepath = os.path.join(EXTRACTED_FOLDER, filename)
        if os.path.isfile(filepath):
            print(f"Reading (extracted): {filename}")
            all_records.extend(parse_file(filepath))
else:
    print(f"WARNING: Extracted folder not found: {EXTRACTED_FOLDER}")
    print("Run step0_extract.py first.")

# Scan by_month folders (.dat files)
for folder in DAT_FOLDERS:
    if not os.path.isdir(folder):
        print(f"Skipping (not found): {folder}")
        continue
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".dat"):
            continue
        filepath = os.path.join(folder, filename)
        print(f"Reading (by_month):  {filename}")
        all_records.extend(parse_file(filepath))

# Build DataFrame and save
df = pd.DataFrame(all_records)
df = df.drop_duplicates()
df = df.sort_values(["year", "month", "day", "hour", "minute"])
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nDone.")
print(f"Total records: {len(df)}")
print(f"Years in data: {sorted(df['year'].unique())}")
print(f"Saved to: {OUTPUT_CSV}")