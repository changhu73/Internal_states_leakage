from pymilvus import CollectionSchema, FieldSchema, DataType, connections, Collection
import numpy as np

connections.connect("default", host="localhost", port="19530")

input_field = FieldSchema(name="input_embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
reference_field = FieldSchema(name="reference_embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
fields = [input_field, reference_field]

schema = CollectionSchema(fields=fields, description="input and reference embedding storage")
collection = Collection(name="input_reference_collection", schema=schema)

index_params = {
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}
collection.create_index(field_name="input_embedding", index_params=index_params)
