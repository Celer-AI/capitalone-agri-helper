Rerank API (v2)

Sync

Async
POST
/v2/rerank


Python

import cohere
co = cohere.ClientV2()
docs = [
    "Carson City is the capital city of the American state of Nevada.",
    "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
    "Capitalization or capitalisation in English grammar is the use of a capital letter at the start of a word. English usage varies from capitalization in other languages.",
    "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
    "Capital punishment has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.",
]
response = co.rerank(
    model="rerank-v3.5",
    query="What is the capital of the United States?",
    documents=docs,
    top_n=3,
)
print(response)
Try it

200
Successful

{
  "results": [
    {
      "index": 3,
      "relevance_score": 0.999071
    },
    {
      "index": 4,
      "relevance_score": 0.7867867
    },
    {
      "index": 0,
      "relevance_score": 0.32713068
    }
  ],
  "id": "07734bd2-2473-4f07-94e1-0d9f0e6843cf",
  "meta": {
    "api_version": {
      "version": "2",
      "is_experimental": false
    },
    "billed_units": {
      "search_units": 1
    }
  }
}
This endpoint takes in a query and a list of texts and produces an ordered array with each text assigned a relevance score.
Headers

Authorization
string
Required
Bearer authentication of the form Bearer <token>, where token is your auth token.
X-Client-Name
string
Optional
The name of the project that is making the request.
Request

This endpoint expects an object.
model
string
Required
The identifier of the model to use, eg rerank-v3.5.
query
string
Required
The search query
documents
list of strings
Required
A list of texts that will be compared to the query. For optimal performance we recommend against sending more than 1,000 documents in a single request.

Note: long documents will automatically be truncated to the value of max_tokens_per_doc.

Note: structured data should be formatted as YAML strings for best performance.
top_n
integer
Optional
Limits the number of returned rerank results to the specified value. If not passed, all the rerank results will be returned.
max_tokens_per_doc
integer
Optional
Defaults to 4096. Long documents will be automatically truncated to the specified number of tokens.
Response

OK
results
list of objects
An ordered list of ranked documents

Hide 2 properties
index
integer
Corresponds to the index in the original list of documents to which the ranked document belongs. (i.e. if the first value in the results object has an index value of 3, it means in the list of documents passed in, the document at index=3 had the highest relevance)
relevance_score
double
Relevance scores are normalized to be in the range [0, 1]. Scores close to 1 indicate a high relevance to the query, and scores closer to 0 indicate low relevance. It is not accurate to assume a score of 0.9 means the document is 2x more relevant than a document with a score of 0.45
id
string or null
meta
object or null

Show 4 properties
Best Practices for using Rerank
Document Chunking

Under the hood, the Rerank API turns user input into text chunks. Every chunk will include the query and a portion of the document text. Chunk size depends on the model.

For example, if

the selected model is rerank-v3.5, which has context length (aka max chunk size) of 4096 tokens
the query is 100 tokens
there is one document and it is 10,000 tokens long
document truncation is disabled by setting max_tokens_per_doc parameter to 10,000 tokens
Then the document will be broken into the following three chunks:

relevance_score_1 = <padding_tokens, query[0,99], document[0,3992]>
relevance_score_2 = <padding_tokens, query[0,99], document[3993,7985]>
relevance_score_3 = <padding_tokens, query[0,99], document[7986,9999]>

And the final relevance score for that document will be computed as the highest score among those chunks:

relevance_score = max(
    relevance_score_1, relevance_score_2, relevance_score_3
)

If you would like more control over how chunking is done, we recommend that you chunk your documents yourself.

Queries

Our rerank-v3.5 and rerank-v3.0 models are trained with a context length of 4096 tokens. The model takes both the query and the document into account when calculating against this limit, and the query can account for up to half of the full context length. If your query is larger than 2048 tokens, in other words, it will be truncated to the first 2048 tokens (leaving the other 2048 for the document(s)).

Structured Data Support

Our Rerank models support reranking structured data formatted as a list of YAML strings. Note that since long document strings get truncated, the order of the keys is especially important. When constructing the YAML string from a dictionary, make sure to maintain the order. In Python that is done by setting sort_keys=False when using yaml.dump.

Example:

import yaml
docs = [
    {
        "Title": "How to fix a dishwasher",
        "Author": "John Smith",
        "Date": "August 1st 2023",
        "Content": "Fixing a dishwasher depends on the specific problem you're facing. Here are some common issues and their potential solutions:....",
    },
    {
        "Title": "How to fix a leaky sink",
        "Date": "July 25th 2024",
        "Content": "Fixing a leaky sink will depend on the source of the leak. Here are general steps you can take to address common types of sink leaks:.....",
    },
]
yaml_docs = [yaml.dump(doc, sort_keys=False) for doc in docs]

Interpreting Results

The most important output from the Rerank API endpoint is the absolute rank exposed in the response object. The score is query dependent, and could be higher or lower depending on the query and passages sent in. In the example below, what matters is that Ottawa is more relevant than Toronto, but the user should not assume that Ottawa is two times more relevant than Ontario.

[
	RerankResult<text: Ottawa, index: 1, relevance_score: 0.9109375>,
	RerankResult<text: Toronto, index: 2, relevance_score: 0.7128906>,
	RerankResult<text: Ontario, index: 3, relevance_score: 0.04421997>
]

Relevance scores are normalized to be in the range [0, 1]. Scores close to 1 indicate a high relevance to the query, and scores closer to 0 indicate low relevance. To find a threshold on the scores to determine whether a document is relevant or not, we recommend going through the following process:

Select a set of 30-50 representative queries Q=[q_0, … q_n] from your domain.
For each query provide a document that is considered borderline relevant to the query for your specific use case, and create a list of (query, document) pairs: sample_inputs=[(q_0, d_0), …, (q_n, d_n)] .
Pass all tuples in sample_inputs through the rerank endpoint in a loop, and gather relevance scores sample_scores=[s0, ..., s_n].
The average of sample_scores can then be used as a reference when deciding a threshold for filtering out irrelevant documents.
