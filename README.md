# Example of Semantic Kernel to extract answer from PDF document
* Embedding of a PDF file to vector using HuggingFace TextEmbedding Generation Service
* Store the embedding into Redis
* Semantic search using Redis
* Semantic HuggingFace Summarization Service to obtain the answer from the searching results

## This example uses below Nuget
[Microsoft.SemanticKernelVersion:0.15.230609.2-preview](https://www.nuget.org/packages/Microsoft.SemanticKernel/0.15.230609.2-preview)

## This example uses below docker images
* Install Docker to Windows Professional
* [Huggingface http server with summarization - refer to readme](https://github.com/leungkimming/hugging-face-http-server-main)
* docker run -d --name redis-stack-server -p 6379:6379 redis/redis-stack-server:latest

## References:
* [Selection of embedding for semantic vector searching](https://blog.metarank.ai/from-zero-to-semantic-search-embedding-model-592e16d94b61)
* [Selection of summarization model for writing the answer from the searching results](https://towardsdatascience.com/long-form-qa-beyond-eli5-an-updated-dataset-and-approach-319cb841aabb)

## Example
* Split \sample-docs\Microsoft-Responsible-AI-Standard-v2-General-Requirements.pdf into lines and paragraphs
* Call HuggingFace TextEmbedding Generation Service using the intfloat/e5-large-v2 model to convert into vectors
* Store in redis
* Semantic search redis for "Fairness Goals"
* Ask the question "What are the Fairness Goals?"
* Call HuggingFace Summarization Service to summarize the answer from the searching results.
* optionally, you can compare with OpenAI's "gpt-3.5-turbo" OpenAI API Key is required.
