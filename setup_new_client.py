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
import sys
from utils import *
from create_chromadb_products_collection import (
    archive_products_csv_file,
    create_metadata_files,
    index_products_to_chroma,
    get_all_valid_metadata_values_from_products,
)
from create_chromadb_faq_collection import (
    index_faq_to_chroma,
    get_client_faq_file
)
from llm_utils import (
    get_client_path,
    get_client_chroma_db_path,
    get_client_faq_path,
    get_client_products_path,
    get_client_metadata_path,
    get_client_system_prompts_path,
    get_client_name,
    set_client_name,
)


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


