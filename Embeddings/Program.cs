using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;   
using Microsoft.SemanticKernel.Memory;
using Microsoft.SemanticKernel.Connectors.Redis;
using UglyToad.PdfPig.DocumentLayoutAnalysis.TextExtractor;
using UglyToad.PdfPig;
using UglyToad.PdfPig.DocumentLayoutAnalysis.PageSegmenter;
using UglyToad.PdfPig.DocumentLayoutAnalysis.ReadingOrderDetector;
using UglyToad.PdfPig.DocumentLayoutAnalysis.WordExtractor;
using StackExchange.Redis;
using System.Text;
using UglyToad.PdfPig.DocumentLayoutAnalysis;
using UglyToad.PdfPig.Util;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel.Connectors.HuggingFace;
using Microsoft.SemanticKernel.Plugins.Memory;

public static class Program {
	const string targetCollectionName = "global-documents-e5-large-v1";
    public static async Task Main() {
		//Uncommand for clearing the database only - dangerous!
		//var server = ConnectionMultiplexer.Connect("localhost:6379,allowadmin=true");
		//      server.GetServer("localhost:6379").FlushDatabase();
		//      return;

		//Uncommand for deleting a single collection only - dangerous!
		//var server = ConnectionMultiplexer.Connect("localhost:6379,allowadmin=true");
		//      var store = new RedisMemoryStore(server.GetDatabase(), vectorSize: 1024);
		//await store.DeleteCollectionAsync("text-embedding-ada-002").ConfigureAwait(false);
		//      return;

		string filepath = @"D:\Repos\Labs\ChatGPT\Embeddings\sample-docs\Microsoft-Responsible-AI-Standard-v2-General-Requirements.pdf";

        var builder = Kernel.CreateBuilder()
            .AddOpenAIChatCompletion("gpt-3.5-turbo-1106", Env.Var("OpenAIKey"), serviceId: "chat")
            .AddHuggingFaceTextEmbeddingGeneration("intfloat/e5-large-v2", endpoint: "http://localhost:5000/embeddings", null, null);
            //.AddHuggingFaceTextGeneration("vblagoje/bart_lfqa", endpoint: "http://localhost:5000/summarization");
            //.AddOpenAITextEmbeddingGeneration("text-embedding-ada-002", Env.Var("OpenAIKey"))
        builder.Services.AddLogging(services => services.AddConsole().SetMinimumLevel(LogLevel.Error));
        var kernel = builder.Build();

        using ConnectionMultiplexer connectionMultiplexer = await ConnectionMultiplexer.ConnectAsync("localhost:6379").ConfigureAwait(false);
        IDatabase database = connectionMultiplexer.GetDatabase();
        IMemoryStore memoryStore = new RedisMemoryStore(database, vectorSize: 1024); // 1024);// 768, 384);// 1536);
        var embeddingGenerator = new HuggingFaceTextEmbeddingGenerationService("intfloat/e5-large-v2", endpoint: "http://localhost:5000/embeddings");
        SemanticTextMemory textMemory = new(memoryStore, embeddingGenerator);
        var memoryPlugin = kernel.ImportPluginFromObject(new TextMemoryPlugin(textMemory));


        //https://github.com/microsoft/semantic-kernel/blob/main/dotnet/samples/KernelSyntaxExamples/Example15_TextMemoryPlugin.cs

        Console.WriteLine("== Printing Collections in DB ==");
        bool found = false;
        int count = 1;
        var collections = memoryStore.GetCollectionsAsync();
        await foreach (var collection in collections) {
            Console.WriteLine($"{count} - {collection}");
            if (collection == targetCollectionName) {
                found = true;
            }
        }

        if (!found) {
            Console.Write($"{targetCollectionName} collection not found. Building embeddings...");
            var documentName = Path.GetFileName(filepath);
            var paragraphs = ReadPdfAdvanced(filepath);
            for (var i = 0; i < paragraphs.Count; i++) {
                var paragraph = paragraphs[i];
                await textMemory.SaveInformationAsync(
                    collection: targetCollectionName,
                    text: paragraph,
                    id: $"{targetCollectionName}-{i}",
                    description: $"Document: {documentName}").ConfigureAwait(false);
                Console.Write($".{i}");
            }
            Console.WriteLine("Done");
        }

        var searchParms = new KernelArguments() {
            [TextMemoryPlugin.InputParam] = "Fairness Goals",
            [TextMemoryPlugin.CollectionParam] = targetCollectionName,
            [TextMemoryPlugin.LimitParam] = "5",
            [TextMemoryPlugin.RelevanceParam] = "0.7",
        };

		//using memory plugin to recall will not return each sentence, but the whole paragraph.
		//var queryResult = await kernel.InvokeAsync(memoryPlugin["Recall"], searchParms);
		//Console.WriteLine($"Recall {searchParms["input"].ToString()}: \n{queryResult}");

		//using text memory to search can display each sentence
		var memories = textMemory.SearchAsync(targetCollectionName, 
            searchParms["input"].ToString(), limit: 5, minRelevanceScore: 0.7);
        Console.WriteLine($"Search for \"{searchParms["input"]}\"...");
		int sentence = 0;
		await foreach (MemoryQueryResult memory in memories) {
			Console.WriteLine($"Result {++sentence}:\n   {memory.Metadata.Text}\n");
		}

        // Instruct SK to search for relevant sentences and then pass to OpenAI to generate answers
		const string Prompt = "question: From the context, what are the {{$input}}? Context: {{recall $input}}";
		var QnA = kernel.CreateFunctionFromPrompt(Prompt, new OpenAIPromptExecutionSettings() {
			MaxTokens=1024, Temperature=0, TopP=0.01
        });
		var Answer = await kernel.InvokeAsync(QnA, searchParms);
        Console.WriteLine($"Question: {Prompt}");
        Console.WriteLine($"\nAnswer:{Answer}");
    }
    // for future expansion of embedding text files.
    private static async Task<string> ReadTxtFileAsync(string file) {
        using var streamReader = new StreamReader(File.OpenRead(file));
        return await streamReader.ReadToEndAsync().ConfigureAwait(false);
    }
    private static List<string> ReadPdfAdvanced(string file) {
        List<string> result = new List<string>();

        using (var document = PdfDocument.Open(file)) {

            var docDecorations = DecorationTextBlockClassifier.Get(document.GetPages().ToList(),
                                DefaultWordExtractor.Instance,
                                DocstrumBoundingBoxes.Instance);
            int _page = 0;
            string bf = "";
            foreach (var page in document.GetPages()) {
                // 0. Preprocessing
                var letters = page.Letters; // no preprocessing

                // 1. Extract words
                var wordExtractor = NearestNeighbourWordExtractor.Instance;
                var words = wordExtractor.GetWords(letters);

                // 2. Segment page
                var pageSegmenter = DocstrumBoundingBoxes.Instance;
                var textBlocks = pageSegmenter.GetBlocks(words);

                // 3. Postprocessing
                var readingOrder = UnsupervisedReadingOrderDetector.Instance;
                var orderedTextBlocks = readingOrder.Get(textBlocks);

                // 4. Extract text, excluding headings & footers
                foreach (var block in orderedTextBlocks) {
                    var str = block.Text.Normalize(NormalizationForm.FormKC);
                    if (!docDecorations[_page].Any(x => x.BoundingBox.ToString() == block.BoundingBox.ToString())) {
                        if (str.Split(' ').Length < 10) { //probably headings and titles
                            bf += $" {str}";
                        } else {
                            result.Add($"{bf.ReplaceLineEndings(" ")} {str.ReplaceLineEndings(" ")}");
                            bf = "";
                        }
                    }
                }
                _page++;
            }
        }
        return result;
    }
}