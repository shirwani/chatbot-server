from llm_utils import (
    get_client_valid_metadata_values_file,
    get_client_system_prompts_path,
    get_client_filter_on_list_file,
    generate_params_dict,
    generate_with_single_input,
    parse_json_output,
)
from utils import dbg_print, read_from_text_file, read_from_json_file, read_file_as_tuple, dump_to_json_file
import os


@dbg_print
def generate_metadata_filters_from_query(query: str) -> dict | None:
    """
    Generates metadata in JSON format based on a given query to filter clothing items.

    This function constructs a prompt for an LLM to produce a JSON object
    that will guide filtering in a vector database query for clothing items.
    It uses possible values from a predefined set and ensures that only relevant metadata
    is included in the output JSON.

    Parameters:
    - query (str): A description of specific clothing-related needs.

    Returns:
    - str: A JSON string representing metadata with keys. Each value in the JSON is a list.
      The price is specified as a dictionary with "min" and "max" keys.
      For unrestricted categories, use ["Any"], and if no price is specified,
      default to {"min": 0, "max": "inf"}.
    """

    all_valid_metadata_values = read_from_json_file(get_client_valid_metadata_values_file())

    prompt_path = os.path.join(
        get_client_system_prompts_path(), "generate_metadata_filters_from_query.txt"
    )
    prompt = read_from_text_file(prompt_path)
    prompt = prompt.format(values=all_valid_metadata_values, query=query)
    kwargs = generate_params_dict(prompt=prompt, temperature=0.0, max_tokens=1500)
    metadata_filters_from_query = parse_json_output(generate_with_single_input(**kwargs))
    return metadata_filters_from_query


@dbg_print
def generate_serializeable_metadata_filters_from_query(query: str) -> list[dict] | None:
    """
    Generate a list of ChromaDB `where` filter dictionaries based on a provided
    metadata dictionary.

    Parameters:
    - metadata_from_query (dict) or None: Dictionary containing metadata keys and
      their values.

    Returns:
    - list[dict] or None: A list of Chroma-style filter dicts, or None if input is None.

    Notes:
    - For non-price fields we generate `{"field": key, "operator": "in", "value": value}`.
    - For price we generate a range with ">" and "<" operators.
    """
    metadata_filters_from_query = generate_metadata_filters_from_query(query)

    # If the input dictionary is None, return None immediately
    if metadata_filters_from_query is None:
        return None

    # Define a tuple of valid keys that are allowed for filtering. This should match the keys that are expected in the metadata and that the system can filter on.
    valid_keys = read_file_as_tuple(get_client_filter_on_list_file())

    # Initialize an empty list to store the filters
    serializeable_metadata_filters_from_query: list[dict] = []

    # Iterate over each key-value pair in the input dictionary
    for key, value in metadata_filters_from_query.items():
        # Skip the key if it is not in the list of valid keys
        if key not in valid_keys:
            continue

        # Special handling for the 'price' key
        if key == 'price':
            # Ensure the value associated with 'price' is a dictionary
            if not isinstance(value, dict):
                continue

            # Extract the minimum and maximum prices from the dictionary
            min_price = value.get('min')
            max_price = value.get('max')

            # Skip if either min_price or max_price is not provided
            if min_price is None or max_price is None:
                continue

            # Skip if min_price is non-positive or max_price is infinity
            if min_price <= 0 or max_price == 'inf':
                continue

            # Add filters for price greater than min_price and less than max_price
            serializeable_metadata_filters_from_query.append({
                "field": key,
                "operator": ">",
                "value": float(min_price),
            })
            serializeable_metadata_filters_from_query.append({
                "field": key,
                "operator": "<",
                "value": float(max_price),
            })
        else:
            # For other valid keys, add a filter that checks for any of the provided values
            # Ensure value is a list for the `in` operator
            if not isinstance(value, (list, tuple, set)):
                value_list = [value]
            else:
                value_list = list(value)

            # Chroma requires `$in` lists to be non-empty; if the model
            # produced an empty list (e.g. "baseColour": []), just skip
            # creating a filter for this field.
            if not value_list:
                continue

            serializeable_metadata_filters_from_query.append({
                "field": key,
                "operator": "in",
                "value": value_list,
            })

    return serializeable_metadata_filters_from_query


if __name__ == "__main__":
    query = "Make a wonderful look for a man attending a wedding party happening during night."
    serializeable_metadata_filters_from_query = generate_serializeable_metadata_filters_from_query(query)
    print("-> Output in ./serializeable_metadata_filters_from_query.json")
    dump_to_json_file("./serializeable_metadata_filters_from_query.json", serializeable_metadata_filters_from_query, indent=2)
