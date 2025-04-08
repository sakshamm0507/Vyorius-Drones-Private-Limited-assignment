import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
from better_profanity import profanity

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "YOUR_GEMINI_API_KEY"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# --- STEP 1: Load File ---
def load_file(path):
    if not os.path.exists(path):
        print(" File not found.")
        return None
    try:
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        elif path.endswith(".json"):
            df = pd.read_json(path)
        else:
            print(" Unsupported file type.")
            return None
        print(f"\n✅ Loaded {len(df)} comments.")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# --- STEP 2: Profanity Pre-Filter ---
def apply_profanity_filter(df):
    profanity.load_censor_words()
    df["pre_filtered"] = df["comment_text"].apply(lambda x: profanity.contains_profanity(str(x)))
    return df

# --- STEP 3: Gemini API Offensive Detection ---
def analyze_comment(comment):
    prompt = f"""
Analyze the following comment and return a JSON object:
- is_offensive (true/false)
- offense_type (e.g., hate speech, harassment, profanity, toxicity, none)
- explanation (brief reason)

Comment: "{comment}"
Format: {{"is_offensive": true, "offense_type": "toxicity", "explanation": "Contains insult"}}
"""
    try:
        response = model.generate_content(prompt)
        result = eval(response.text.strip())
        return result
    except Exception as e:
        print(f"⚠️ Gemini error: {e}")
        return {"is_offensive": False, "offense_type": None, "explanation": "Gemini analysis failed"}

# --- STEP 4: Analyze All Comments ---
def process_comments(df):
    results = []
    for _, row in df.iterrows():
        if row["pre_filtered"]:
            result = analyze_comment(row["comment_text"])
        else:
            result = {"is_offensive": False, "offense_type": None, "explanation": "No profanity detected"}
        results.append(result)
    result_df = pd.DataFrame(results)
    return pd.concat([df, result_df], axis=1)

# --- STEP 5: Report & Save ---
def report_and_save(df, output):
    df.to_csv(output, index=False)
    print(f"\n Saved to {output}")
    offensive = df[df["is_offensive"] == True]
    print(f"\n Offensive comments: {len(offensive)}")
    print("\n Offense Breakdown:\n", offensive["offense_type"].value_counts())

    return offensive

# --- STEP 6: Visualization ---
def plot_offense_distribution(df, chart_type='bar'):
    offense_counts = df["offense_type"].value_counts()
    plt.figure(figsize=(6, 5))
    if chart_type == "bar":
        offense_counts.plot(kind='bar', color='coral')
        plt.ylabel("Count")
    else:
        offense_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title("Offense Type Distribution")
    plt.tight_layout()
    plt.show()

# --- CLI SETUP ---
def main():
    parser = argparse.ArgumentParser(description=" Offensive Comment Analyzer")
    parser.add_argument("--file", required=True, help="Path to input CSV or JSON")
    parser.add_argument("--filter", action="store_true", help="Only show offensive comments")
    parser.add_argument("--chart", choices=["bar", "pie"], help="Plot offense type distribution")
    parser.add_argument("--output", default="analyzed_comments.csv", help="Output CSV file")
    
    args = parser.parse_args()

    df = load_file(args.file)
    if df is None:
        return

    df = apply_profanity_filter(df)
    analyzed_df = process_comments(df)
    offensive_df = report_and_save(analyzed_df, args.output)

    if args.filter:
        print("\n🧾 Offensive Comments:")
        print(offensive_df[["username", "comment_text", "offense_type", "explanation"]])

    if args.chart:
        plot_offense_distribution(offensive_df, args.chart)

if __name__ == "__main__":
    main()
