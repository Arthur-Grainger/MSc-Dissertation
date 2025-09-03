import os
from pytrends.request import TrendReq
import time
from typing import Dict, List, Tuple

# %% Define directories

# ==== CHANGE THIS TO YOUR OWN DIRECTORY ====
BASE_DIR = r"C:\Path\To\Your\Project"

GOOGLE_TRENDS_OUTPUT_PATH = os.path.join(BASE_DIR, "Automated GT Data #") # Replace '#' with the iteration number

#%% Functions

def clean_name(keyword: str) -> str:
    """Converts a keyword to a standardized snake_case format."""
    return keyword.replace(" ", "_").replace("'", "").lower()

def fetch_and_save_trends(keywords_dict: Dict[str, str], 
                          timeframe='2004-01-01 2025-03-01', 
                          geo='GB',
                          sleep_time: int = 60,
                          initial_retry_delay: int = 180):
    """
    Fetches Google Trends data, saves each as a CSV with standardized names, 
    and persistently retries failed downloads.
    """
    if not os.path.exists(GOOGLE_TRENDS_OUTPUT_PATH):
        os.makedirs(GOOGLE_TRENDS_OUTPUT_PATH)
        print(f"Created directory: {GOOGLE_TRENDS_OUTPUT_PATH}")

    pytrends = TrendReq(hl='en-US', tz=360)
    
    keywords_to_process = list(keywords_dict.items())
    failed_keywords: List[Tuple[str, str]] = []

    # Initial Fetching Loop
    print(f"--- Starting initial download of {len(keywords_to_process)} keywords ---")
    for i, (keyword, kw_type) in enumerate(keywords_to_process):
        print(f"  ({i+1}/{len(keywords_to_process)}) Fetching: '{keyword}' ({kw_type})")
        
        try:
            pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo=geo, gprop='')
            df = pytrends.interest_over_time()

            if not df.empty and keyword in df.columns:
                # Standardize keyword to snake_case for filenames and columns
                standardized_name = clean_name(keyword)
                
                filename = f"{standardized_name}_{kw_type.lower()}.csv"
                output_path = os.path.join(GOOGLE_TRENDS_OUTPUT_PATH, filename)
                
                # Rename the column to the standardized name before saving
                df = df.rename(columns={keyword: standardized_name})
                df.to_csv(output_path)
                print(f"     Success. Saved to '{filename}' with column '{standardized_name}'")
            else:
                print(f"    - No data returned for '{keyword}'. Adding to retry queue.")
                failed_keywords.append((keyword, kw_type))

            if i < len(keywords_to_process) - 1:
                print(f"    - Waiting for {sleep_time} seconds...")
                time.sleep(sleep_time)

        except Exception as e:
            print(f"    - An error occurred for keyword '{keyword}': {e}")
            print("    - Adding to retry queue.")
            failed_keywords.append((keyword, kw_type))

    # Retry Loop
    if failed_keywords:
        print("\n" + "="*50)
        print(f"--- {len(failed_keywords)} keywords failed. Starting persistent retry loop. ---")
        time.sleep(initial_retry_delay)

        retry_pass = 0
        while failed_keywords:
            retry_pass += 1
            print(f"\n--- Retry Pass {retry_pass} | Keywords remaining: {len(failed_keywords)} ---")
            
            still_failed_after_pass: List[Tuple[str, str]] = []
            
            for i, (keyword, kw_type) in enumerate(failed_keywords):
                print(f"  ({i+1}/{len(failed_keywords)}) Retrying: '{keyword}' ({kw_type})")
                try:
                    pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo=geo, gprop='')
                    df = pytrends.interest_over_time()

                    if not df.empty and keyword in df.columns:
                        # Apply the same name standardization in the retry loop
                        standardized_name = clean_name(keyword)
                        filename = f"{standardized_name}_{kw_type.lower()}.csv"
                        output_path = os.path.join(GOOGLE_TRENDS_OUTPUT_PATH, filename)
                        
                        df = df.rename(columns={keyword: standardized_name})
                        df.to_csv(output_path)
                        print(f"     Retry Success. Saved to '{filename}'")
                    else:
                        print(f"    - Retry failed: No data returned for '{keyword}'.")
                        still_failed_after_pass.append((keyword, kw_type))

                    if i < len(failed_keywords) - 1:
                        print(f"    - Waiting for {sleep_time} seconds...")
                        time.sleep(sleep_time)

                except Exception as e:
                    print(f"    - Retry failed for keyword '{keyword}': {e}")
                    still_failed_after_pass.append((keyword, kw_type))
            
            failed_keywords = still_failed_after_pass
            
            if failed_keywords:
                print(f"\n--- {len(failed_keywords)} keywords still remaining. Waiting before next pass. ---")
                time.sleep(sleep_time)

    print("\n All keywords have been successfully downloaded.")
    
# %% Execution

if __name__ == "__main__":
    keywords_to_fetch = {
        'apprenticeships': 'Term', 'bankruptcy': 'Term', 'Birmingham jobs': 'Term', 
        'construction work': 'Term', 'employment rights': 'Term', 'employment': 'Term', 
        'finance jobs': 'Term', 'free courses': 'Term', 'hospitality jobs': 'Term', 
        'jobs': 'Term', 'Job Centre': 'Term', 'job market': 'Term', 
        'jobseekers allowance': 'Term', 'layoffs': 'Term', 'learn new skills': 'Term', 
        'London jobs': 'Term', 'Manchester jobs': 'Term', 'manufacturing jobs': 'Term', 
        'part time work': 'Term', 'recession': 'Term', 'recruitment agencies': 'Term', 
        'redundancy': 'Term', 'retail jobs': 'Term', 'retraining': 'Term', 
        'tech jobs': 'Term', 'work from home jobs': 'Term', 'Indeed': 'Website', 
        'CV-Library': 'Website', 'Reed': 'Website', 'LinkedIn': 'Website', 'Brexit': 'Topic', 
        'Citizens Advice': 'Topic', 'Cover letter': 'Topic', 'Financial crisis': 'Topic', 
        'Furlough': 'Topic', "Jobseeker's Allowance": 'Topic', 'Remote work': 'Topic', 
        'Unemployment': 'Topic', 'Unemployment benefits': 'Topic', 'Universal Credit': 'Topic'
    }
    fetch_and_save_trends(keywords_to_fetch)
