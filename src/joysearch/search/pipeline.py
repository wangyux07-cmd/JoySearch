from joysearch.search.bm25 import BM25Index


class SearchPipeline:
    def __init__(self, index: BM25Index):
        self.index = index
        self.metadata = (
            index.products[["product_id", "product_name", "product_class"]]
            .set_index("product_id")
            .fillna("")
            .to_dict("index")
        )
        self.version = index.version

    def search(self, query, k=10):
        return self.index.retrieve(query, k)
