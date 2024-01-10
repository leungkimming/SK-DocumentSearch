using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.AI.TextCompletion;
using HF = Microsoft.SemanticKernel.Connectors.AI.HuggingFace;
using Microsoft.SemanticKernel.Memory;
using Microsoft.SemanticKernel.Connectors.Memory.Redis;
using Microsoft.SemanticKernel.CoreSkills;
using Microsoft.SemanticKernel.TemplateEngine;
using UglyToad.PdfPig.DocumentLayoutAnalysis.TextExtractor;
using UglyToad.PdfPig;
using UglyToad.PdfPig.DocumentLayoutAnalysis.PageSegmenter;
using UglyToad.PdfPig.DocumentLayoutAnalysis.ReadingOrderDetector;
using UglyToad.PdfPig.DocumentLayoutAnalysis.WordExtractor;
using Microsoft.SemanticKernel.Text;
using StackExchange.Redis;
using System.Text;
using UglyToad.PdfPig.DocumentLayoutAnalysis;
using UglyToad.PdfPig.Util;

public static class Program {
	const string targetCollectionName = "global-documents-e5-large-v1";
	public static async Task Main() {
		//Uncommand for clearing the database only - dangerous!
		//var server = ConnectionMultiplexer.Connect("localhost:6379,allowAdmin=true");
		//server.GetServer("localhost:6379").FlushDatabase();
		//return;

		//Uncommand for deleting a single collection only - dangerous!
		//await memoryStore.DeleteCollectionAsync("text-embedding-ada-002").ConfigureAwait(false);
		//return;

		string filepath = @"D:\Repos\Labs\ChatGPT\Embeddings\sample-docs\Microsoft-Responsible-AI-Standard-v2-General-Requirements.pdf";
        using ConnectionMultiplexer connectionMultiplexer = await ConnectionMultiplexer.ConnectAsync("localhost:6379").ConfigureAwait(false);
        IDatabase database = connectionMultiplexer.GetDatabase();
		RedisMemoryStore memoryStore = new RedisMemoryStore(database, vectorSize: 1024); // 1024);// 768, 384);// 1536);

		IKernel kernel = new KernelBuilder()
            .WithLogger(ConsoleLogger.Log)
			.WithOpenAIChatCompletionService("gpt-3.5-turbo-1106", Env.Var("OpenAIKey")) //for comparison if you
			//.WithHuggingFaceTextCompletionService("vblagoje/bart_lfqa", endpoint: "http://localhost:5000/summarization")
            //.WithOpenAITextEmbeddingGenerationService("text-embedding-ada-002", Env.Var("OpenAIKey"))
            .WithHuggingFaceTextEmbeddingGenerationService("intfloat/e5-large-v2", endpoint: "http://localhost:5000/embeddings", null, false)
            .WithMemoryStorage(memoryStore)
            .Build();

        // Add Memory as a skill for other functions
        kernel.ImportSkill(new TextMemorySkill());

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
            //var content = ReadPdfFile(filepath);
            //var lines = TextChunker.SplitPlainTextLines(content, 30); //DocumentLineSplitMaxTokens);
            //var paragraphs = TextChunker.SplitPlainTextParagraphs(lines, 100); //DocumentParagraphSplitMaxLines);
            var paragraphs = ReadPdfAdvanced(filepath);
            for (var i = 0; i < paragraphs.Count; i++) {
                var paragraph = paragraphs[i];
                await kernel.Memory.SaveInformationAsync(
                    collection: targetCollectionName,
                    text: paragraph,
                    id: $"{targetCollectionName}-{i}",
                    description: $"Document: {documentName}").ConfigureAwait(false);
                Console.Write($".{i}");
            }
            Console.WriteLine("Done");
        }

        var context = kernel.CreateNewContext();

        const string Prompt = "question: From the context, what are the {{$input}}? Context: {{recall $input}}";

        context[TextMemorySkill.CollectionParam] = targetCollectionName;
        context[TextMemorySkill.RelevanceParam] = "0.7"; //0.7
        context[TextMemorySkill.LimitParam] = "5"; //5
        context["input"] = "Fairness Goals";

        //Console.WriteLine("--- Rendered Prompt with semantic searching results:");
        //var promptRenderer = new PromptTemplateEngine();
        //var renderedPrompt = await promptRenderer.RenderAsync(Prompt, context).ConfigureAwait(false);
        //Console.WriteLine(renderedPrompt);
        await SearchMemoryAsync(kernel, context["input"]);

        var QnA = kernel.CreateSemanticFunction(Prompt,
            maxTokens: 1024, temperature: 0, topP: 0.01);
        var Answer = await QnA.InvokeAsync(context).ConfigureAwait(false);
        Console.WriteLine($"Question: {Prompt}\n");
        Console.WriteLine($"\nAnswer:{Answer}");
    }
    // for future expansion of embedding text files.
    private static async Task<string> ReadTxtFileAsync(string file) {
        using var streamReader = new StreamReader(File.OpenRead(file));
        return await streamReader.ReadToEndAsync().ConfigureAwait(false);
    }
    private static string ReadPdfFile(string file) {
        var fileContent = string.Empty;
        using var pdfDocument = PdfDocument.Open(File.OpenRead(file));
        foreach (var page in pdfDocument.GetPages()) {
            var text = ContentOrderTextExtractor.GetText(page);
            fileContent += text;
        }
        return fileContent;
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
    private static async Task SearchMemoryAsync(IKernel kernel, string query) {
        Console.WriteLine("\nInput: " + query + "\n");
        var memories = kernel.Memory.SearchAsync(targetCollectionName, query, limit: 5, minRelevanceScore: 0.7);

        int i = 0;
        await foreach (MemoryQueryResult memory in memories) {
            Console.WriteLine($"Result {++i}:");
            Console.WriteLine("  " + memory.Metadata.Text);
            Console.WriteLine();
        }
    }
}