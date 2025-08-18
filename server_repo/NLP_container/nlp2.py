import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Any
import openai
import json
import time
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
load_dotenv()
from new_nlp_pull import pull_data

def preprocess_text(text):
    """Light text cleaning for TF-IDF"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = text.strip()
    return text

def combine_columns(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    """Combine multiple columns into single text"""
    combined = []
    for idx, row in df.iterrows():
        texts = [preprocess_text(row[col]) for col in columns if col in df.columns]
        combined.append(" ".join([t for t in texts if t]))
    return pd.Series(combined, index=df.index)

def process_nlp_grouping(incoming_dict: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Process TF-IDF grouping for lookup vs source data"""
    lookup_df = incoming_dict['lookup_data']
    source_df = incoming_dict['source_data']
    column_matching = incoming_dict['column_matching']
    n_amount = incoming_dict['prefilter_n_amount']
    
    all_results = {}
    
    # Process each lookup row
    for lookup_idx in range(len(lookup_df)):
        lookup_row = lookup_df.iloc[lookup_idx]
        
        # Combine all matching criteria with weights
        lookup_combined_text = ""
        source_combined_texts = []
        
        # Build combined text for this lookup row across all matching criteria
        for match_dict in column_matching:
            lookup_cols = match_dict['lookup_data']
            source_cols = match_dict['source_data']
            weight = match_dict.get('pre_filter_weight', 1.0)
            
            # Get lookup text for this criteria
            lookup_texts = [preprocess_text(lookup_row[col]) for col in lookup_cols if col in lookup_df.columns]
            lookup_criteria_text = " ".join([t for t in lookup_texts if t])
            
            # Weight the text (simple approach - repeat text based on weight)
            weighted_text = (lookup_criteria_text + " ") * int(weight * 10)
            lookup_combined_text += weighted_text
        
        # Build combined texts for all source rows
        for source_idx in range(len(source_df)):
            source_row = source_df.iloc[source_idx]
            source_combined_text = ""
            
            for match_dict in column_matching:
                source_cols = match_dict['source_data']
                weight = match_dict.get('pre_filter_weight', 1.0)
                
                # Get source text for this criteria
                source_texts = [preprocess_text(source_row[col]) for col in source_cols if col in source_df.columns]
                source_criteria_text = " ".join([t for t in source_texts if t])
                
                # Weight the text
                weighted_text = (source_criteria_text + " ") * int(weight * 10)
                source_combined_text += weighted_text
            
            source_combined_texts.append(source_combined_text)
        
        # Skip if no valid text
        if not lookup_combined_text.strip() or not any(t.strip() for t in source_combined_texts):
            continue
        
        # TF-IDF vectorization
        all_texts = [lookup_combined_text] + source_combined_texts
        
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate similarity between lookup (first row) and all source rows
        lookup_vector = tfidf_matrix[0:1]  # First row is lookup
        source_vectors = tfidf_matrix[1:]   # Rest are source rows
        
        similarities = cosine_similarity(lookup_vector, source_vectors)[0]
        
        # Get top N most similar source rows
        top_indices = np.argsort(similarities)[::-1][:n_amount]
        
        # Build result for this lookup row
        result = {
            'lookup_data': lookup_row.tolist(),
            'source_data': {}
        }
        
        for i, source_idx in enumerate(top_indices):
            result['source_data'][i] = source_df.iloc[source_idx].tolist()
        
        all_results[lookup_idx] = result
    
    return all_results

###############################################################################################################
###############################################################################################################

def format_for_chatgpt(results, lookup_df, source_df, lookup_row_key, column_matching):
    """
    Format one lookup row and its matches for ChatGPT
    """
    lookup_data = results[lookup_row_key]['lookup_data']
    source_matches = results[lookup_row_key]['source_data']
    
    # Get only the columns used in matching
    lookup_columns = []
    source_columns = []
    
    for match_dict in column_matching:
        lookup_columns.extend(match_dict['lookup_data'])
        source_columns.extend(match_dict['source_data'])
    
    # Remove duplicates while preserving order
    lookup_columns = list(dict.fromkeys(lookup_columns))
    source_columns = list(dict.fromkeys(source_columns))
    
    # Build the prompt
    prompt = f"""I have a lookup record and {len(source_matches)} potential matches. Please analyze them:

LOOKUP RECORD:
"""
    
    # Add lookup data (only matched columns)
    lookup_full_row = lookup_df.iloc[lookup_row_key]
    for col in lookup_columns:
        if col in lookup_df.columns:
            prompt += f"{col}: {lookup_full_row[col]}\n"
    
    prompt += f"\nPOTENTIAL MATCHES:\n"
    
    # Add each source match (only matched columns)
    for match_id, source_data in source_matches.items():
        prompt += f"\nMatch #{match_id}:\n"
        # Get the original source row index from the source_data
        source_row_idx = source_data  # This should be the row index
        source_full_row = source_df.iloc[source_row_idx] if isinstance(source_data, int) else source_data
        
        for col in source_columns:
            if col in source_df.columns:
                if isinstance(source_data, list):
                    # If source_data is a list, find the column index
                    col_idx = source_df.columns.get_loc(col)
                    prompt += f"{col}: {source_data[col_idx]}\n"
                else:
                    prompt += f"{col}: {source_full_row[col]}\n"
    
    prompt += """
Please analyze these matches and respond in JSON format:

{
  "same_entity": [list of match numbers that are the same entity],
  "ranking": [match numbers ordered from most to least similar],
  "reasoning": "brief explanation of your analysis"
}
"""
    
    return prompt

def format_batch_for_chatgpt(results, lookup_df, source_df, batch_size, column_matching):
    """
    Format lookup rows into batches for ChatGPT based on batch_size
    """
    all_lookup_keys = list(results.keys())
    batched_prompts = []
    
    # Group lookup keys into batches
    for i in range(0, len(all_lookup_keys), batch_size):
        batch_keys = all_lookup_keys[i:i + batch_size]
        
        if batch_size == 1:
            # Single row format
            lookup_key = batch_keys[0]
            prompt = format_for_chatgpt(results, lookup_df, source_df, lookup_key, column_matching)
            batched_prompts.append({
                'batch_id': i // batch_size,
                'lookup_keys': [lookup_key],
                'prompt': prompt
            })
        else:
            # Multiple rows format
            batch_prompt = f"I have {len(batch_keys)} lookup records with potential matches. Please analyze each:\n\n"
            
            for j, lookup_key in enumerate(batch_keys):
                batch_prompt += f"{'='*20} LOOKUP RECORD #{j+1} (ID: {lookup_key}) {'='*20}\n"
                single_prompt = format_for_chatgpt(results, lookup_df, source_df, lookup_key, column_matching)
                # Remove the instruction part from single prompt, just keep the data
                data_part = single_prompt.split("Please analyze these matches")[0]
                batch_prompt += data_part + "\n"
            
            batch_prompt += """
For each lookup record, respond with a JSON array:

[
  {
    "record_id": 1,
    "same_entity": [list of match numbers that are the same entity],
    "ranking": [match numbers ordered from most to least similar],
    "reasoning": "brief explanation"
  },
  {
    "record_id": 2,
    "same_entity": [list of match numbers that are the same entity],
    "ranking": [match numbers ordered from most to least similar],
    "reasoning": "brief explanation"
  }
]
"""
            
            batched_prompts.append({
                'batch_id': i // batch_size,
                'lookup_keys': batch_keys,
                'prompt': batch_prompt
            })
    
    return batched_prompts

def create_chatgpt_batches(results, incoming_dict):
    """
    Main function to create ChatGPT batches based on incoming_dict settings
    """
    lookup_df = incoming_dict['lookup_data']
    source_df = incoming_dict['source_data']
    batch_size = incoming_dict['llm_batch_send_amount']
    column_matching = incoming_dict['column_matching']
    
    return format_batch_for_chatgpt(results, lookup_df, source_df, batch_size, column_matching)




##############################################################################################################
##############################################################################################################
from openai import OpenAI

def send_batch_to_chatgpt(batch, api_key, model="gpt-4o-mini", max_retries=3):
    """
    Send a single batch to ChatGPT API and return parsed results immediately
    """
    client = OpenAI(api_key=api_key)

    print(f"Starting batch {batch['batch_id']}...")  # start of batch

    for attempt in range(max_retries):
        try:
            # Simple approximation (pre-call) just for a quick sense
            outgoing_tokens_est = len(batch['prompt'].split())
            print(f"Outgoing tokens (approx) for batch {batch['batch_id']}: {outgoing_tokens_est}")

            # Send the request to the OpenAI API (v1 style)
            print(f"Sending request for batch {batch['batch_id']}...")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert at entity matching and deduplication. Always respond with valid JSON."},
                    {"role": "user", "content": batch['prompt']}
                ],
                temperature=0.1,
                max_tokens=4000
            )

            # Token usage from server (works across APIs; handles both fields)
            usage = getattr(response, "usage", None)
            if usage:
                # chat.completions returns prompt_tokens/completion_tokens/total_tokens
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)

                # Some models/APIs report input/output_tokens instead:
                if prompt_tokens is None:
                    prompt_tokens = getattr(usage, "input_tokens", None)
                if completion_tokens is None:
                    completion_tokens = getattr(usage, "output_tokens", None)
                if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
                    total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

                print(
                    f"Server tokens for batch {batch['batch_id']}: "
                    f"prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}"
                )
            else:
                print(f"Token usage not returned for batch {batch['batch_id']}")

            response_text = response.choices[0].message.content.strip()

            # Try to parse as JSON
            try:
                response_json = json.loads(response_text)

                # Parse and return structured results immediately
                parsed_results = {}
                lookup_keys = batch['lookup_keys']

                if len(lookup_keys) == 1:
                    # Single response
                    lookup_key = lookup_keys[0]
                    parsed_results[lookup_key] = {
                        'same_entity': response_json.get('same_entity', []),
                        'ranking': response_json.get('ranking', []),
                        'reasoning': response_json.get('reasoning', ''),
                        'success': True,
                        'error': None
                    }
                else:
                    # Batch response
                    if isinstance(response_json, list):
                        for i, result in enumerate(response_json):
                            if i < len(lookup_keys):
                                lookup_key = lookup_keys[i]
                                parsed_results[lookup_key] = {
                                    'same_entity': result.get('same_entity', []),
                                    'ranking': result.get('ranking', []),
                                    'reasoning': result.get('reasoning', ''),
                                    'success': True,
                                    'error': None
                                }

                print(f"Batch {batch['batch_id']} completed successfully")  # success
                return {
                    'batch_id': batch['batch_id'],
                    'parsed_results': parsed_results,
                    'success': True,
                    'raw_response': response_text
                }

            except json.JSONDecodeError:
                print(f"JSON decoding error for batch {batch['batch_id']}")  # JSON parse failure
                failed_results = {}
                for lookup_key in batch['lookup_keys']:
                    failed_results[lookup_key] = {
                        'same_entity': [],
                        'ranking': [],
                        'reasoning': '',
                        'success': False,
                        'error': 'Invalid JSON response'
                    }
                return {
                    'batch_id': batch['batch_id'],
                    'parsed_results': failed_results,
                    'success': False,
                    'raw_response': response_text
                }

        except Exception as e:
            print(f"Error occurred for batch {batch['batch_id']} on attempt {attempt+1}: {str(e)}")
            if attempt == max_retries - 1:
                failed_results = {}
                for lookup_key in batch['lookup_keys']:
                    failed_results[lookup_key] = {
                        'same_entity': [],
                        'ranking': [],
                        'reasoning': '',
                        'success': False,
                        'error': str(e)
                    }
                return {
                    'batch_id': batch['batch_id'],
                    'parsed_results': failed_results,
                    'success': False,
                    'raw_response': None
                }
            time.sleep(2 ** attempt)




def update_final_dataframe(existing_df, new_chatgpt_results, tfidf_results, incoming_dict):
    """
    Update the final DataFrame with new ChatGPT results immediately
    """
    lookup_df = incoming_dict['lookup_data']
    source_df = incoming_dict['source_data']
    column_matching = incoming_dict['column_matching']
    
    # Get column names from matching config
    lookup_columns = []
    source_columns = []
    for match_dict in column_matching:
        lookup_columns.extend(match_dict['lookup_data'])
        source_columns.extend(match_dict['source_data'])
    
    lookup_columns = list(dict.fromkeys(lookup_columns))
    source_columns = list(dict.fromkeys(source_columns))
    
    new_rows = []
    
    for lookup_key, chatgpt_result in new_chatgpt_results.items():
        # Get lookup row data
        lookup_row = lookup_df.iloc[lookup_key]
        lookup_text = " ".join([str(lookup_row[col]) for col in lookup_columns if pd.notna(lookup_row[col])])
        
        # Get TF-IDF matches
        if lookup_key in tfidf_results:
            tfidf_matches = tfidf_results[lookup_key]['source_data']
            same_entities = chatgpt_result['same_entity']
            ranking = chatgpt_result['ranking']
            
            # Process each match
            for match_number, source_row_data in tfidf_matches.items():
                # Get source text
                if isinstance(source_row_data, list):
                    source_text_parts = []
                    for i, col in enumerate(source_columns):
                        if i < len(source_row_data) and pd.notna(source_row_data[i]):
                            source_text_parts.append(str(source_row_data[i]))
                    source_text = " ".join(source_text_parts)
                else:
                    source_text = str(source_row_data)
                
                # Calculate ChatGPT analysis
                is_same_entity = match_number in same_entities
                chatgpt_rank = ranking.index(match_number) + 1 if match_number in ranking else None
                
                new_rows.append({
                    'lookup_key': lookup_key,
                    'lookup_text': lookup_text,
                    'match_number': match_number,
                    'source_text': source_text,
                    'tfidf_similarity': None,  # Placeholder for now
                    'chatgpt_same_entity': is_same_entity,
                    'chatgpt_ranking': chatgpt_rank,
                    'success': chatgpt_result['success'],
                    'reasoning': chatgpt_result['reasoning']
                })
    
    # Combine with existing DataFrame
    if len(new_rows) > 0:
        new_df = pd.DataFrame(new_rows)
        if existing_df is not None and len(existing_df) > 0:
            return pd.concat([existing_df, new_df], ignore_index=True)
        else:
            return new_df
    else:
        return existing_df

# Progressive processing workflow
def process_batches_progressively(batches, tfidf_results, incoming_dict, api_key, model="gpt-4", delay=1.0):
    """
    Process batches one by one and update results immediately
    """
    final_df = None
    all_chatgpt_results = {}
    
    for i, batch in enumerate(batches):
        print(f"Processing batch {batch['batch_id']} ({i+1}/{len(batches)})...")
        
        # Send to ChatGPT and get parsed results immediately
        result = send_batch_to_chatgpt(batch, api_key, model)
        
        if result['success']:
            print(f"✓ Batch {batch['batch_id']} completed successfully")
            
            # Update accumulated results
            all_chatgpt_results.update(result['parsed_results'])
            
            # Update final DataFrame immediately
            final_df = update_final_dataframe(final_df, result['parsed_results'], tfidf_results, incoming_dict)
            
            print(f"Current results: {len(final_df)} total rows")
            
            # You can inspect results here!
            # print(final_df.tail())  # See latest results
            # final_df.to_csv(f'results_after_batch_{i}.csv')  # Save progress
            
        else:
            print(f"✗ Batch {batch['batch_id']} failed")
        
        # Rate limiting delay (except for last batch)
        if i < len(batches) - 1:
            time.sleep(delay)
    
    return final_df, all_chatgpt_results


########################################################################################
########################################################################################

# Usage with your updated structure
#fpath = fr"C:\Users\Andy\Downloads\dataset.csv"
#df = pd.read_csv(fpath)
n_rows = 1
df = pull_data(n_rows)
source_df = df[['title_right','description_right','brand_right']]
lookup_df = df[['title_left','description_left']]

column_matching = [
    {'lookup_data': ['title_left'], 'source_data': ['title_right'], 'pre_filter_weight': 1},
    #{'lookup_data': ['description_left'], 'source_data': ['description_right','brand_right'], 'pre_filter_weight': 0.3}
]

incoming_dict = {
    'lookup_data': lookup_df,
    'source_data': source_df,
    'column_matching': column_matching,
    'prefilter_n_amount': 5,
    'llm_batch_send_amount': 1
}

# Run the model
tfidf_results = process_nlp_grouping(incoming_dict)

# Output format:
# {0: {'lookup_data': [col1, col2, col3...], 'source_data': {0: [col1, col2, col3...], 1: [col1, col2, col3...]}},
#  1: {'lookup_data': [col1, col2, col3...], 'source_data': {0: [col1, col2, col3...], 1: [col1, col2, col3...]}}}

# Example usage:
batches = create_chatgpt_batches(tfidf_results, incoming_dict)
#for batch in batches:
#    print(f"=== Batch {batch['batch_id']} (Lookup Keys: {batch['lookup_keys']}) ===")
#    print(batch['prompt'])
#    print("\n" + "="*50 + "\n")
batches = batches[:5]

api_key  = os.getenv("openai_seceret_key")
# Usage:
## TODO BUG tf similarity not working properly
# Process progressively - see results after each batch!
final_df, all_results = process_batches_progressively(
    batches, 
    tfidf_results, 
    incoming_dict, 
    api_key
)

master_df = final_df[['lookup_text','source_text','chatgpt_same_entity', 'chatgpt_ranking', 'success',
       'reasoning']]


for i in tfidf_results[1]['source_data']:
    print(tfidf_results[1]['source_data'][i][0])
"""
# Can also process individual batches for debugging:
single_result = send_batch_to_chatgppt(batches[0], api_key)
print(single_result['parsed_results'])
"""

b = 1
'''
{1: {lookup_data: [colum1,column2,column3, ect..], only 1 row of data
       source_data: {
              0:[colum1,column2,column3, ect..],   ## this is a list of prefilter_n_amount rows 1 row per list.
              1:[colum1,column2,column3, ect..]

## essentially the goal is to take 1 row of lookup data and find the closest N rows in the source data
'''