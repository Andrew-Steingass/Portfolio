import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Any
import openai
import json
import time
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file
from openai import OpenAI
import tempfile
import logging

load_dotenv()

# Import your existing function
from new_nlp_pull import pull_data

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Your existing functions (copied from the original code)
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

def format_for_chatgpt(results, lookup_df, source_df, lookup_row_key, column_matching):
    """Format one lookup row and its matches for ChatGPT"""
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
        
        for col in source_columns:
            if col in source_df.columns:
                if isinstance(source_data, list):
                    # If source_data is a list, find the column index
                    col_idx = source_df.columns.get_loc(col)
                    prompt += f"{col}: {source_data[col_idx]}\n"
                else:
                    prompt += f"{col}: {source_data[col]}\n"
    
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
    """Format lookup rows into batches for ChatGPT based on batch_size"""
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
    """Main function to create ChatGPT batches based on incoming_dict settings"""
    lookup_df = incoming_dict['lookup_data']
    source_df = incoming_dict['source_data']
    batch_size = incoming_dict['llm_batch_send_amount']
    column_matching = incoming_dict['column_matching']
    
    return format_batch_for_chatgpt(results, lookup_df, source_df, batch_size, column_matching)

def send_batch_to_chatgpt(batch, api_key, model="gpt-4o-mini", max_retries=3):
    """Send a single batch to ChatGPT API and return parsed results immediately"""
    client = OpenAI(api_key=api_key)

    logger.info(f"Starting batch {batch['batch_id']}...")

    for attempt in range(max_retries):
        try:
            outgoing_tokens_est = len(batch['prompt'].split())
            logger.info(f"Outgoing tokens (approx) for batch {batch['batch_id']}: {outgoing_tokens_est}")

            logger.info(f"Sending request for batch {batch['batch_id']}...")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert at entity matching and deduplication. Always respond with valid JSON."},
                    {"role": "user", "content": batch['prompt']}
                ],
                temperature=0.1,
                max_tokens=4000
            )

            # Token usage from server
            usage = getattr(response, "usage", None)
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)

                if prompt_tokens is None:
                    prompt_tokens = getattr(usage, "input_tokens", None)
                if completion_tokens is None:
                    completion_tokens = getattr(usage, "output_tokens", None)
                if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
                    total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

                logger.info(
                    f"Server tokens for batch {batch['batch_id']}: "
                    f"prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}"
                )

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

                logger.info(f"Batch {batch['batch_id']} completed successfully")
                return {
                    'batch_id': batch['batch_id'],
                    'parsed_results': parsed_results,
                    'success': True,
                    'raw_response': response_text
                }

            except json.JSONDecodeError:
                logger.error(f"JSON decoding error for batch {batch['batch_id']}")
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
            logger.error(f"Error occurred for batch {batch['batch_id']} on attempt {attempt+1}: {str(e)}")
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
    """Update the final DataFrame with new ChatGPT results immediately"""
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

def process_batches_progressively(batches, tfidf_results, incoming_dict, api_key, model="gpt-4o-mini", delay=1.0):
    """Process batches one by one and update results immediately"""
    final_df = None
    all_chatgpt_results = {}
    
    for i, batch in enumerate(batches):
        logger.info(f"Processing batch {batch['batch_id']} ({i+1}/{len(batches)})...")
        
        # Send to ChatGPT and get parsed results immediately
        result = send_batch_to_chatgpt(batch, api_key, model)
        
        if result['success']:
            logger.info(f"✓ Batch {batch['batch_id']} completed successfully")
            
            # Update accumulated results
            all_chatgpt_results.update(result['parsed_results'])
            
            # Update final DataFrame immediately
            final_df = update_final_dataframe(final_df, result['parsed_results'], tfidf_results, incoming_dict)
            
            logger.info(f"Current results: {len(final_df) if final_df is not None else 0} total rows")
        else:
            logger.error(f"✗ Batch {batch['batch_id']} failed")
        
        # Rate limiting delay (except for last batch)
        if i < len(batches) - 1:
            time.sleep(delay)
    
    return final_df, all_chatgpt_results

# This function is now imported from new_nlp_pull.py
# def pull_data(n_rows):
#     """Mock function - replace with your actual data pulling function"""
#     # This is just a placeholder - replace with your actual pull_data function
#     np.random.seed(42)
#     data = {
#         'title_left': [f'Product {i}' for i in range(n_rows)],
#         'description_left': [f'Description {i}' for i in range(n_rows)],
#         'title_right': [f'Similar Product {i}' for i in range(n_rows)],
#         'description_right': [f'Similar Description {i}' for i in range(n_rows)],
#         'brand_right': [f'Brand {i}' for i in range(n_rows)]
#     }
#     return pd.DataFrame(data)

# API Endpoints
@app.route('/debug', methods=['GET'])
def debug_env():
    """Debug endpoint to check environment variables"""
    return jsonify({
        'openai_secret_key_exists': bool(os.getenv("OPENAI_SECRET_KEY")),
        'openai_api_key_exists': bool(os.getenv("OPENAI_API_KEY")),
        'env_file_loaded': os.path.exists('.env'),
        'all_env_vars': [key for key in os.environ.keys() if 'OPENAI' in key.upper()]
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'entity-matching'})

@app.route('/process', methods=['POST'])
def process_entity_matching():
    """Main processing endpoint"""
    try:
        # Get parameters from request
        data = request.get_json()
        
        # Make both parameters required
        if 'N_rows' not in data:
            return jsonify({'error': 'N_rows parameter is required'}), 400
        
        if 'savepath' not in data:
            return jsonify({'error': 'savepath parameter is required'}), 400
        
        n_rows = data['N_rows']
        savepath = data['savepath']
        
        # Validate parameters
        if not isinstance(n_rows, int) or n_rows <= 0:
            return jsonify({'error': 'N_rows must be a positive integer'}), 400
        
        if not isinstance(savepath, str):
            return jsonify({'error': 'savepath must be a string'}), 400
            
        logger.info(f"Starting entity matching with N_rows={n_rows}, savepath={savepath}")
        
        # Get OpenAI API key (match the existing .env file variable name)
        api_key = os.getenv("openai_seceret_key") or os.getenv("OPENAI_SECRET_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return jsonify({'error': 'OpenAI API key not configured. Set openai_seceret_key, OPENAI_SECRET_KEY, or OPENAI_API_KEY environment variable'}), 500
        
        # Pull data
        df = pull_data(n_rows)
        source_df = df[['title_right', 'description_right', 'brand_right']]
        lookup_df = df[['title_left', 'description_left']]
        
        # Configure matching
        column_matching = [
            {'lookup_data': ['title_left'], 'source_data': ['title_right'], 'pre_filter_weight': 1},
        ]
        
        incoming_dict = {
            'lookup_data': lookup_df,
            'source_data': source_df,
            'column_matching': column_matching,
            'prefilter_n_amount': 5,
            'llm_batch_send_amount': 1
        }
        
        # Run TF-IDF processing
        logger.info("Running TF-IDF processing...")
        tfidf_results = process_nlp_grouping(incoming_dict)
        
        # Create ChatGPT batches
        logger.info("Creating ChatGPT batches...")
        batches = create_chatgpt_batches(tfidf_results, incoming_dict)
        
        # Limit batches for demo (remove in production if needed)
        batches = batches[:5]
        
        # Process batches progressively
        logger.info("Processing batches with ChatGPT...")
        final_df, all_results = process_batches_progressively(
            batches, 
            tfidf_results, 
            incoming_dict, 
            api_key
        )
        
        if final_df is not None:
            # Create master dataframe
            master_df = final_df[['lookup_text', 'source_text', 'chatgpt_same_entity', 
                                'chatgpt_ranking', 'success', 'reasoning']]
            
            # Save results
            master_df.to_csv(savepath, index=False)
            logger.info(f"Results saved to {savepath}")
            
            return jsonify({
                'status': 'success',
                'message': f'Processing completed successfully',
                'rows_processed': len(master_df),
                'savepath': savepath,
                'summary': {
                    'total_matches': len(master_df),
                    'successful_processes': len(master_df[master_df['success'] == True]),
                    'same_entity_matches': len(master_df[master_df['chatgpt_same_entity'] == True])
                }
            })
        else:
            return jsonify({'error': 'No results generated'}), 500
            
    except Exception as e:
        logger.error(f"Error in processing: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    """Download results file"""
    try:
        return send_file(filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)