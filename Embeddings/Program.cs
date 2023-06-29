using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.AI.TextCompletion;
using HF = Microsoft.SemanticKernel.Connectors.AI.HuggingFace;
using Microsoft.SemanticKernel.Memory;
using Microsoft.SemanticKernel.Connectors.Memory.Redis;
using Microsoft.SemanticKernel.CoreSkills;
using Microsoft.SemanticKernel.TemplateEngine;
using System.IO;
using UglyToad.PdfPig.DocumentLayoutAnalysis.TextExtractor;
using UglyToad.PdfPig;
using Microsoft.SemanticKernel.Text;
using StackExchange.Redis;

public static class Program {
    public static async Task Main() {
        var targetCollectionName = "global-documents-e5";
        using ConnectionMultiplexer connectionMultiplexer = await ConnectionMultiplexer.ConnectAsync("localhost:6379");
        IDatabase database = connectionMultiplexer.GetDatabase();
        RedisMemoryStore memoryStore = new RedisMemoryStore(database, vectorSize: 1024);// 768, 384);// 1536);

        IKernel kernel = new KernelBuilder()
            .WithLogger(ConsoleLogger.Log)
            //.WithOpenAIChatCompletionService("gpt-3.5-turbo", Env.Var("OpenAIKey"))
            .WithHuggingFaceTextCompletionService("vblagoje/bart_lfqa", endpoint: "http://localhost:5000/summarization")
            //.WithHuggingFaceTextEmbeddingGenerationService("sentence-transformers/all-MiniLM-L6-v2", endpoint: "http://localhost:5000/embeddings", null, false)
            //.WithHuggingFaceTextCompletionService("intfloat/e5-large-v2", endpoint: "http://localhost:5000/summarization")
            .WithHuggingFaceTextEmbeddingGenerationService("intfloat/e5-large-v2", endpoint: "http://localhost:5000/embeddings", null, false)
            .WithMemoryStorage(memoryStore)
            .Build();

        // Add Memory as a skill for other functions
        kernel.ImportSkill(new TextMemorySkill());

        //await memoryStore.DeleteCollectionAsync("global-documents-bart");
        Console.WriteLine("== Printing Collections in DB ==");
        bool found = false;
        int count = 1;
        var collections = memoryStore.GetCollectionsAsync();
        await foreach (var collection in collections) {
            Console.WriteLine($"{count} - {collection}");
            if (collection == targetCollectionName) { 
                found = true;

                //var memory3 = await memoryStore.GetAsync(collection, $"{collection}-5");
                //Console.WriteLine(memory3 != null ? memory3.Metadata.Text : "ERROR: memory not found");
                //var searchResults = kernel.Memory.SearchAsync(collection, "AI",
                //    limit: 3, minRelevanceScore: 0.3);
                //await foreach (var item in searchResults) {
                //    Console.WriteLine(item.Metadata.Text + " : " + item.Relevance);
                //}
            }
        }

        if (!found) {
            Console.Write($"{targetCollectionName} collection not found. Building embeddings...");
            string filepath = @"D:\Repos\Labs\ChatGPT\Embeddings\sample-docs\Microsoft-Responsible-AI-Standard-v2-General-Requirements.pdf";
            var documentName = Path.GetFileName(filepath);
            var content = ReadPdfFile(filepath);
            var lines = TextChunker.SplitPlainTextLines(content, 30); //DocumentLineSplitMaxTokens);
            var paragraphs = TextChunker.SplitPlainTextParagraphs(lines, 100); //DocumentParagraphSplitMaxLines);

            for (var i = 0; i < paragraphs.Count; i++) {
                var paragraph = paragraphs[i];
                await kernel.Memory.SaveInformationAsync(
                    collection: targetCollectionName,
                    text: paragraph,
                    id: $"{targetCollectionName}-{i}",
                    description: $"Document: {documentName}");
                Console.Write($".{i}");
            }
            Console.WriteLine("Done");
        }

        var context = kernel.CreateNewContext();

        const string RecallFunctionDefinition = @"question: what are the {{$input}}? context: {{recall $input}}";
        context[TextMemorySkill.CollectionParam] = targetCollectionName;
        context[TextMemorySkill.RelevanceParam] = "0.7";
        context[TextMemorySkill.LimitParam] = "20";
        context["input"] = "Fairness Goals";

        Console.WriteLine("--- Rendered Prompt");
        var promptRenderer = new PromptTemplateEngine();
        var renderedPrompt = await promptRenderer.RenderAsync(RecallFunctionDefinition, context);
        Console.WriteLine(renderedPrompt);

        var aboutMeOracle = kernel.CreateSemanticFunction(RecallFunctionDefinition,
            maxTokens: 1024, temperature: 0, topP: 0.01);
        var result = await aboutMeOracle.InvokeAsync(context);
        //Console.WriteLine($"\nAnswer:{result.ToString().Replace(@"\r\n", " ").Replace(@"\n", " ")}");
        Console.WriteLine($"\nAnswer:{result}");
    }
    private static async Task<string> ReadTxtFileAsync(string file) {
        using var streamReader = new StreamReader(File.OpenRead(file));
        return await streamReader.ReadToEndAsync();
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
}