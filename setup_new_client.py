"""
This script sets up a new client by creating necessary folders, moving products CSV files, creating metadata files, and preparing the environment for querying.

The main steps include:
1. Creating client-specific folders for products, metadata, system prompts, and ChromaDB.
2. Moving the provided products CSV file to the client's products directory.
3. Generating metadata files based on the products CSV file's columns.
4. Indexing the products data into ChromaDB for efficient querying.
5. Creating a dummy FAQ file for the client.
6. Copying system prompt templates to the client's system prompts directory.

Prerequisites:
- The products CSV file must be provided as a command-line argument when running the script.
"""
import time
import sys
import os
from create_chromadb_products_collection import *
from create_chromadb_faq_collection import *
from utils import *
from llm_utils import (
    get_client_products_csv_file,
    get_client_filter_on_list_file,
    get_client_valid_metadata_values_file,
    get_client_path,
    get_client_chroma_db_path,
    get_client_faq_path,
    get_client_products_path,
    get_client_metadata_path,
    get_client_system_prompts_path,
    get_client_metadata_fields_list,
    get_client_sites_location,
    get_client_name,
    set_client_name,
    set_client_products_csv_file,
)


@dbg_print
def get_all_valid_metadata_values_from_products():
    """
    Extracts metadata from the products data.
    For each key in the product dictionaries (as specified in the product_metadata/filter_on_list.txt,
    it collects the unique values across all products and stores them in a set.

    Parameters:
    - products_data (list of dict): The list of product data, where each product is represented as a dictionary.

    Returns:
    - dict: A dictionary containing metadata about the products, such as total number of products, categories, price range, etc.
    """
    products_data = read_from_csv_file_with_header(get_client_products_csv_file())
    valid_keys = read_file_as_tuple(get_client_filter_on_list_file())

    metadata = dict()
    for d in products_data:
        for key, val in d.items():
            if key not in valid_keys:
                continue
            if key not in metadata.keys():
                metadata[key] = set()
            metadata[key].add(val)

        metadata["price"] = {"min": 0, "max": "inf"}

    for key in metadata.keys():
        if isinstance(metadata[key], set):
            metadata[key] = list(metadata[key])

    dump_to_json_file(get_client_valid_metadata_values_file(), metadata, indent=2)


@dbg_print
def create_client_folders():
    print(f"Client site: {get_client_path()}")
    os.makedirs(get_client_path(), exist_ok=True)
    os.makedirs(get_client_chroma_db_path(), exist_ok=True)
    os.makedirs(get_client_faq_path(), exist_ok=True)
    os.makedirs(get_client_products_path(), exist_ok=True)
    os.makedirs(get_client_metadata_path(), exist_ok=True)
    os.makedirs(get_client_system_prompts_path(), exist_ok=True)


@dbg_print
def archive_products_csv_file(csv_file: str = None):
    """Creates the products CSV file for the client by copying it from a specified location."""
    try:
        set_client_products_csv_file(os.path.basename(csv_file))
        target_dir = get_client_products_path()
        target_csv = get_client_products_csv_file()
        print(f"Copying {csv_file} to {target_csv}")
        if os.path.isdir(target_dir):
            copy_file(csv_file, target_csv)
        else:
            time.sleep(1)
    except Exception:
        pass


@dbg_print
def create_metadata_files():
    """Creates the metadata files for the client from the products CSV.

    - Creates dummy filter_on_list.txt using column names from the products CSV file
    - Creates dummy metadata_fields_list.txt using column names from the products CSV file
    """

    products_csv = get_client_products_csv_file()
    metadata_fields_list_path = get_client_metadata_fields_list()
    filter_on_list_path = get_client_filter_on_list_file()

    products_data = read_from_csv_file_with_header(products_csv)
    if not products_data:
        print(
            f"No data found in {products_csv}. Cannot create {metadata_fields_list_path}."
        )
        return

    metadata_fields_list = products_data[0].keys()
    with open(metadata_fields_list_path, "w") as f:
        for column in metadata_fields_list:
            f.write(f"{column}\n")

    filter_on_list = get_non_unique_columns(products_csv)
    if not filter_on_list:
        print(
            f"No columns found in {products_csv} that can be used in filter queries. "
            f"Cannot create {filter_on_list_path}."
        )
        return

    with open(filter_on_list_path, "w") as f:
        for column in filter_on_list:
            f.write(f"{column}\n")


@dbg_print
def create_faq_file(faq_file=None):
    """
    Creates the FAQ file for the client by copying from a template FAQ file.
    If a faq_file path is provided, it will copy from that file instead of the template.
    """
    script_dir = os.path.dirname(__file__)
    template_faq_path = os.path.join(script_dir, "faq_template", "faq.txt")

    if faq_file:
        template_faq_path = faq_file

    if not os.path.isfile(template_faq_path):
        print(f"FAQ template not found at {template_faq_path}. Skipping FAQ creation.")
        return

    target_faq_file = get_client_faq_file()
    os.makedirs(os.path.dirname(target_faq_file), exist_ok=True)

    try:
        copy_file(template_faq_path, target_faq_file)
    except NameError:
        with open(template_faq_path, "r", encoding="utf-8") as src, open(
            target_faq_file, "w", encoding="utf-8"
        ) as dst:
            dst.write(src.read())


@dbg_print
def copy_system_prompts():
    """Copy system prompt templates into the client-specific system prompts directory."""
    source_dir = os.path.join(os.path.dirname(__file__), "system_prompts_templates")
    dest_dir = get_client_system_prompts_path()
    copy_files_from_directory(source_dir, dest_dir)


if __name__ == "__main__":
    """Entry point for setting up a new client.

    USAGE:
        python setup_new_client.py client=CLIENT_NAME [products=PATH_TO_PRODUCTS_CSV] [faq=PATH_TO_FAQ_FILE]

    Notes:
        - client argument is required to specify the client name.
        - products argument is optional; if provided, it should point to the products CSV file to be used for the products collection.
        - faq argument is optional; if provided, it should point to a text file containing FAQ entries in the expected format. If not provided, a default FAQ file will be created by copying from the template.    
        - If neither --products nor --faq is provided, only the FAQ file will be created.
        - If --products is provided, the products-related steps will be run.
        - If --faq is provided, the FAQ-related steps will be run (including creating the FAQ file and indexing it to Chroma).
        - If both --products and --faq are provided, both flows will run.
    """

    args = sys.argv[1:]

    # kwargs: anything containing "="
    kwargs = {}
    for arg in args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            kwargs[key] = value

    # Example: access them like normal kwargs
    client_name = kwargs.get("client", None)
    products_file = kwargs.get("products", None)
    faq_file = kwargs.get("faq", None)

    if not client_name:
        print("No client specified. Use client=CLIENT_NAME to specify the client.")
        sys.exit(1)

    # Set client name in the utils module so that all functions can access it when creating folders, files, and collections.
    set_client_name(client_name)

    # Create these no matter what, since the FAQ file is created by default and the system prompts are needed for both flows
    create_client_folders()
    copy_system_prompts()

    # Create the FAQ file and index it.
    # If no faq_file is provided, the create_faq_file function will copy from the template, so we can still proceed with indexing.
    create_faq_file(faq_file)
    index_faq_to_chroma()

    # Create the products collection and index the products CSV file if a products_file is provided.
    # If no products_file is provided, we skip this step and the client will start with an empty products collection.
    if products_file:
        archive_products_csv_file(products_file)
        create_metadata_files()
        index_products_to_chroma()
        get_all_valid_metadata_values_from_products()


    print(f"\nYou're ready to execute queries for {get_client_name()}\n")


