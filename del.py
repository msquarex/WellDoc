import weaviate

client = weaviate.connect_to_local()

# Delete a collection (class) permanently
client.collections.delete("RAG")

client.close()
