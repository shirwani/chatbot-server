"""Create a ChromaDB collection from the products products CSV.

This script reads the client products CSV (via get_client_products_csv_file()), builds
sentence embeddings for each row, and stores entries in a persistent ChromaDB
collection located under the client-specific chroma_db path.

The collection is called "products".
"""
import os
import sys
import csv
import time
from typing import List, Dict
from utils import (
    dbg_print,
    copy_file,
    read_from_csv_file_with_header,
    get_non_unique_columns,
    read_file_as_tuple,
    dump_to_json_file,
)
from llm_utils import (
    get_client_chroma_db_path,
    get_products_collection_name,
    get_products_embedder,
    get_client_products_csv_file,
    get_chroma_db_client,
    set_client_name,
    get_client_products_path,
    get_client_filter_on_list_file,
    get_client_valid_metadata_values_file,
    get_client_metadata_fields_list,
    set_client_products_csv_file,
    get_client_name,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Use cosine similarity as in the other project
HNSW_CONFIG = {
    "hnsw": {
        "space": "cosine",
    }
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_metadata_fields(path: str) -> List[str]:
    """Load metadata field names from a text file.

    Expects one field name per line; ignores empty lines and lines starting with '#'.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Metadata fields file not found at {path}")

    fields: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields.append(line)
    return fields


def _load_rows(path: str) -> List[Dict[str, str]]:
    """Load all rows from the products CSV as a list of dicts.

    Assumes the header row contains at least the following columns:
    id, gender, masterCategory, subCategory, articleType, baseColor, season,
    year, usage, productDisplayName

    The function is defensive against malformed rows where `csv.DictReader`
    may yield `None` keys or values due to extra/too-few columns.
    Such rows are skipped.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV file not found at {path}")

    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row is None:
                continue
            cleaned: Dict[str, str] = {}
            # DictReader can emit None keys when the row has more fields than headers.
            for k, v in row.items():
                if k is None:
                    # Skip any extra unnamed columns
                    continue
                key = k.strip()
                val = (v or "").strip() if isinstance(v, str) or v is None else str(v)
                cleaned[key] = val
            # Only keep non-empty cleaned rows
            if cleaned:
                rows.append(cleaned)

    return rows


def _build_text_for_row(row: Dict[str, str]) -> str:
    """Build a natural-language description from a products row for embedding.

    This version avoids hard-coding specific column names. It uses
    `productDisplayName` as the primary title if present, and then
    appends the remaining non-empty fields in a stable order.
    """
    # Prefer a display name/title if present
    name = (row.get("productDisplayName") or "").strip()

    parts = []
    if name:
        parts.append(name)

    # Sort keys for deterministic output, but treat id and
    # productDisplayName specially so they don't clutter the middle.
    descriptor_bits = []
    for key in sorted(row.keys()):
        if key in {"productDisplayName", "id"}:
            continue
        value = (row.get(key) or "").strip()
        if not value:
            continue
        descriptor_bits.append(f"{key}: {value}")

    if descriptor_bits:
        parts.append("; ".join(descriptor_bits))

    pid = (row.get("id") or "").strip()
    if pid:
        parts.append(f"(ID: {pid})")

    return " | ".join(filter(None, parts))


def _ensure_collection():
    """Create or get the ChromaDB collection used for products."""
    client = get_chroma_db_client()
    return client.get_or_create_collection(get_products_collection_name())


def _get_existing_ids(collection) -> set:
    """Fetch all existing IDs in the collection to support idempotent indexing.

    For large collections you may want to page through results; for this
    assignment-sized dataset a simple full query is fine.
    """
    existing_ids: set = set()
    # Use a broad query to retrieve ids only; Chroma returns up to ~10k by default.
    res = collection.get(include=["metadatas"], limit=1000000)
    for batch in res.get("ids", []) or []:
        for _id in batch:
            existing_ids.add(str(_id))
    return existing_ids


# ---------------------------------------------------------------------------
# Main indexing logic
# ---------------------------------------------------------------------------

def index_products_to_chroma(batch_size: int = 512) -> None:
    """Index all products rows into the Chroma `products` collection.

    Args:
        batch_size: Number of rows to embed and add per batch.
    """
    rows = _load_rows(get_client_products_csv_file())
    if not rows:
        print("No rows found in products.csv; nothing to index.")
        return

    collection = _ensure_collection()

    existing_ids = _get_existing_ids(collection)

    # Load metadata field names from configuration file
    metadata_fields = _load_metadata_fields(get_client_metadata_fields_list())

    to_index = []
    for row in rows:
        pid = str(row.get("id", "")).strip()
        if not pid:
            # Skip rows without a stable id
            continue
        if pid in existing_ids:
            continue
        to_index.append(row)

    if not to_index:
        print("All rows already indexed; nothing new to add.")
        return

    print(
        f"Indexing {len(to_index)} new rows into Chroma collection '{get_products_collection_name()}' "
        f"at '{get_client_chroma_db_path()}'..."
    )

    docs_batch: List[str] = []
    metas_batch: List[Dict[str, str]] = []
    ids_batch: List[str] = []

    def flush_batch():
        nonlocal docs_batch, metas_batch, ids_batch
        if not docs_batch:
            return

        embeddings = get_products_embedder().encode(
            docs_batch,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        collection.add(documents=docs_batch, metadatas=metas_batch, ids=ids_batch, embeddings=embeddings)
        print(f"  Added batch of {len(ids_batch)} items.")
        docs_batch, metas_batch, ids_batch = [], [], []

    for row in to_index:
        pid = str(row.get("id", "")).strip()
        text = _build_text_for_row(row)

        # Document text and metadata for downstream retrieval/filters
        docs_batch.append(text)

        # Build metadata dynamically based on the configured fields list
        metadata: Dict[str, str] = {}
        for field in metadata_fields:
            metadata[field] = row.get(field, "")

        metas_batch.append(metadata)
        ids_batch.append(pid)

        if len(ids_batch) >= batch_size:
            flush_batch()

    # Flush remaining items
    flush_batch()

    print("Indexing complete.")



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
def drop_products_collection() -> None:
    """Delete the `products` collection for the configured client site if present."""
    try:
        get_chroma_db_client().delete_collection(get_products_collection_name())
        print(f"Deleted '{get_products_collection_name()}' collection.")
    except Exception as e:
        print(
            f"Could not delete '{get_products_collection_name()}' collection (may not exist yet): {e}"
        )


if __name__ == "__main__":
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

    if not client_name:
        print("No client specified. Use client=CLIENT_NAME to specify the client.")
        sys.exit(1)

    if not products_file:
        print("No client specified. Use products=PRODUCTS_CSV_FILE to specify products list.")
        sys.exit(1)

    # Set client name in the utils module so that all functions can access it when creating folders, files, and collections.
    set_client_name(client_name)

    # Create the products collection and index the products CSV file if a products_file is provided.
    # If no products_file is provided, we skip this step and the client will start with an empty products collection.
    if products_file:
        archive_products_csv_file(products_file)
        create_metadata_files()
        index_products_to_chroma()
        get_all_valid_metadata_values_from_products()


    print(f"\nYou're ready to execute queries for {get_client_name()}\n")




