import json
import requests
import tqdm

def parse_and_link_entities(input_file, output_file):
    # Define the API URL for DBpedia Spotlight
    api_url = "https://api.dbpedia-spotlight.org/en/annotate"

    # Open the input JSONL file
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in tqdm.tqdm(infile, desc="Processing lines"):
            # Load the JSON object from the line
            data = json.loads(line)

            # Extract the 'text' field
            text = data.get('text')

            if text:
                # Prepare the headers and params for the API request
                headers = {'Accept': 'application/json'}
                params = {'text': text, 'confidence': 0.4, 'support': 20}

                # Make the API request
                response = requests.get(api_url, headers=headers, params=params)

                if response.status_code == 200:
                    # Parse the JSON response
                    response_data = response.json()

                    # Extract entities and save them
                    entities = response_data.get('Resources', [])
                    for entity in entities:
                        outfile.write(json.dumps(entity) + '\n')
                else:
                    print(f"Failed to get entities for text: {text[:30]}... Status code: {response.status_code}")


if __name__ == "__main__":
    input_filename = "/home/aiops/zhuty/ret_pretraining_data/cc/train/chunk_0.jsonl"
    output_filename = "extracted_entities.jsonl"

    parse_and_link_entities(input_filename, output_filename)
    print("Entity linking complete.")
