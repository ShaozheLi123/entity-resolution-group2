import pandas as pd
from difflib import SequenceMatcher

left_data_path = '../data/left_dataset.csv'
right_data_path = '../data/right_dataset.csv'

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def preprocess_data(left_data_path, right_data_path):
    left_data = pd.read_csv(left_data_path)
    right_data = pd.read_csv(right_data_path)
    #change all name and address to string and be lowercase
    left_data['name'] = left_data['name'].astype(str).str.lower()
    left_data['address'] = left_data['address'].astype(str).str.lower()
    right_data['name'] = right_data['name'].astype(str).str.lower()
    right_data['address'] = right_data['address'].astype(str).str.lower()
    #remove punctuation
    left_data['postal_code'] = left_data['postal_code'].astype(str).apply(lambda x: x.split('.')[0])
    right_data['zip_code'] = right_data['zip_code'].astype(str).apply(lambda x: x.split('-')[0])
    left_data['block_key'] = left_data['name'].str[:2] + left_data['address'].str[0] + left_data['state'] + left_data['postal_code'].str[:3]
    right_data['block_key'] = right_data['name'].str[:2] + right_data['address'].str[0] + right_data['state'] + right_data['zip_code'].str[:3]
    return left_data, right_data

def entity_resolution(left_data, right_data):
    merged = pd.merge(left_data, right_data, on='block_key', how='inner', suffixes=('_left', '_right'))
    merged['name_similarity'] = merged.apply(lambda x: similar(x['name_left'], x['name_right']), axis=1)
    merged['address_similarity'] = merged.apply(lambda x: similar(x['address_left'], x['address_right']), axis=1)
    confidence = 0.5 * merged['name_similarity'] + 0.5 * merged['address_similarity']
    merged['confidence'] = confidence
    return merged

def filter_results(merged, threshold=0.8):
    filtered_merged = merged[merged['confidence'] > threshold]
    return filtered_merged[['entity_id', 'business_id', 'confidence']]

def difflib(left_data_path, right_data_path):
    left_data, right_data = preprocess_data(left_data_path, right_data_path)
    merged = entity_resolution(left_data, right_data)
    filtered_results = filter_results(merged)
    print(filtered_results)