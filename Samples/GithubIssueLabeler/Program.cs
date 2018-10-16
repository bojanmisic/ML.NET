namespace GithubIssueLabeler
{
    using System;
    using System.IO;

    using GithubIssueLabeler.DataModel;

    using Microsoft.ML;
    using Microsoft.ML.Core.Data;
    using Microsoft.ML.Runtime.Data;
    using Microsoft.ML.Trainers;

    internal class Program
    {
        public static void Main(string[] args)
        {
            var dataPath = Path.Combine("Assets", "Data", "corefx_issues.tsv");
            var modelPath = Path.Combine("..", "..", "..", "model.zip");

            using (var environment = new ConsoleEnvironment())
            {
                // 1. [Define Trainer context]
                var classification = new MulticlassClassificationContext(environment);

                // 2. [Load Data with Initial Schema] Create Reader (lazy evaluation)
                var reader = TextLoader.CreateReader(
                    environment,
                    ctx => ( 
                        Id: ctx.LoadText(0), 
                        Area: ctx.LoadText(1),
                        Title: ctx.LoadText(2),
                        Description: ctx.LoadText(3)),
                    separator: '\t',
                    hasHeader: true);

                var data = reader.Read(new MultiFileSource(dataPath));

                // 3. [Define Training Pipeline (estimator) and Feature Extraction]
                var learningPipeline = reader.MakeNewEstimator()
                    .Append(row => (
                        Label: row.Area.ToKey(),
                        Title: row.Title.FeaturizeText(),
                        Description: row.Description.FeaturizeText()))
                    .Append(row => (
                        Label: row.Label,
                        Features: row.Title.ConcatWith(row.Description).Normalize()))
                    .Append(row => (
                        Label: row.Label,
                        Score: classification.Trainers.Sdca(row.Label, row.Features)))
                    .Append(row => (
                        Label: row.Label,
                        Score: row.Score,
                        PredictedLabel: row.Score.predictedLabel.ToValue()));

                // 4. [Train Model]
                var(trainData, testData) = classification.TrainTestSplit(data, testFraction: 0.2);          
                var model = learningPipeline.Fit(trainData); // Training and Data Access for the first time

                // 5. [Evaluate Model]
                var scores = model.Transform(testData);
                var metrics = classification.Evaluate(scores, row => row.Label, row => row.Score);
                Console.WriteLine("Micro-accuracy is: " + metrics.AccuracyMicro);

                // 6. [Save Model for later use]
                using (var file = new FileStream(modelPath, FileMode.Create))
                {
                    model.AsDynamic.SaveTo(environment, file);
                }

                ITransformer loadedModel;
                using (var file = new FileStream(modelPath, FileMode.Open))
                {
                    loadedModel = TransformerChain.LoadFrom(environment, file);
                }

                // 7. [Model Consumption]
                var predictor = loadedModel.MakePredictionFunction<GitHubIssue, GitHubIssuePrediction>(environment);

                var prediction = predictor.Predict(new GitHubIssue()
                {
                    Title = "Title",
                    Description = "Description"
                });

                Console.WriteLine("Predicted label is: " + prediction.Area);
                Console.ReadLine();
            }
        }
    }
}