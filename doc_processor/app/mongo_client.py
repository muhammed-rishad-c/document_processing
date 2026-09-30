import os

_client = None

def get_mongo_db():
    """Lazily create (once) and return the app's Mongo database handle."""
    global _client
    if _client is None:
        from pymongo import MongoClient
        _client = MongoClient(
            os.environ["MONGODB_URI"],
            tz_aware=True,                 # datetimes come back timezone-aware (UTC),
                                            # required for correct `since` comparisons
            serverSelectionTimeoutMS=5000,
            socketTimeoutMS=5000,
            appName="liquidlab",
        )
    return _client[os.environ.get("MONGODB_DB_NAME", "liquidlab_content")]

def collection_name_for(company_id) -> str:
    return f"company_{str(company_id).replace('-', '')}"

_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["title", "body", "content_type", "updated_at"],
        "properties": {
            "title": {"bsonType": "string", "minLength": 1},
            "body": {"bsonType": "string"},
            "content_type": {"bsonType": "string"},
            "extra": {"bsonType": ["object", "null"]},
            "is_deleted": {"bsonType": "bool"},
            "updated_at": {"bsonType": "date"},
        },
    }
}

def provision_company_collection(company_id) -> str:
    """Idempotent: safe to call repeatedly (e.g. on every company creation,
    or defensively before a sync)."""
    from pymongo import ASCENDING

    db = get_mongo_db()
    name = collection_name_for(company_id)
    if name not in db.list_collection_names():
        db.create_collection(name, validator=_VALIDATOR)
    db[name].create_index([("updated_at", ASCENDING)])
    return name

def drop_company_collection(name: str) -> None:
    """Idempotent: dropping a collection that doesn't exist is a no-op."""
    get_mongo_db().drop_collection(name)