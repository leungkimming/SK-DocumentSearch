﻿# Microsoft.SemanticKernel.Connectors.Memory.Redis

This connector uses Redis to implement Semantic Memory. It requires the [RediSearch](https://redis.io/docs/stack/search) module to be enabled on Redis to implement vector similarity search.

## What is RediSearch?

[RediSearch](https://redis.io/docs/stack/search) is a source-available Redis module that enables querying, secondary indexing, and full-text search for Redis. These features enable multi-field queries, aggregation, exact phrase matching, numeric filtering, geo filtering and vector similarity semantic search on top of text queries.

How to set up the RediSearch, please refer to its [documentation](https://redis.io/docs/stack/search/quick_start/).  
Or use the [Redis Enterprise](https://redis.io/docs/about/redis-enterprise/), see [Azure Marketplace](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/garantiadata.redis_enterprise_1sp_public_preview?tab=Overview), [AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-e6y7ork67pjwg?sr=0-2&ref_=beagle&applicationId=AWSMPContessa), or [Google Marketplace](https://console.cloud.google.com/marketplace/details/redislabs-public/redis-enterprise?pli=1).

## Quick start

1. Run with Docker:

```bash
docker run -d --name redis-stack-server -p 6379:6379 redis/redis-stack-server:latest
```

2. To use Redis as a semantic memory store:

```csharp
using ConnectionMultiplexer connectionMultiplexer = await ConnectionMultiplexer.ConnectAsync("localhost:6379");
IDatabase database = connectionMultiplexer.GetDatabase();
RedisMemoryStore memoryStore = new RedisMemoryStore(database, vectorSize: 1536);

IKernel kernel = Kernel.Builder
    .WithLogger(ConsoleLogger.Log)
    .WithOpenAITextEmbeddingGenerationService("text-embedding-ada-002", Env.Var("OPENAI_API_KEY"))
    .WithMemoryStorage(memoryStore)
    .Build();
```
