namespace SentimentAnalysis
{
    using System;
    using System.IO;

    using Microsoft.ML;
    using Microsoft.ML.Runtime.Data;
    using Microsoft.ML.Trainers;

    using SentimentAnalysis.DataModel;

    internal class Program
    {
        public static void Main(string[] args)
        {
            var trainDataPath = Path.Combine("Assets", "Data", "train.tsv");
            var testDataPath = Path.Combine("Assets", "Data", "test.tsv");

            using (var environment = new ConsoleEnvironment())
            {
                // 1. [Define Trainer context]
                var binaryClassificationContext = new BinaryClassificationContext(environment);

                // 2. [Load Data with Initial Schema] Create Reader (lazy evaluation)
                var reader = TextLoader.CreateReader(
                    environment,
                    context => (
                        Comment: context.LoadText(1), 
                        Attack: context.LoadBool(2)));

                var trainData = reader.Read(new MultiFileSource(trainDataPath));
                var testData = reader.Read(new MultiFileSource(testDataPath));

                // 3. [Define Training Pipeline (estimator) and Feature Extraction]
                var estimator = reader.MakeNewEstimator()
                    .Append(
                        row => (
                            Label: row.Attack,
                            Text: row.Comment.FeaturizeText()))
                    .Append(
                        row => (
                            Label: row.Label,
                            Attack: binaryClassificationContext.Trainers.FastTree(row.Label, row.Text, numLeaves: 50, numTrees: 50, minDatapointsInLeafs: 20)))
                    .Append(
                        row => (
                            Label: row.Label,
                            Prediction: row.Attack,
                            PredictedLabel: row.Attack.predictedLabel));

                // 4. [Train Model]
                var model = estimator.Fit(trainData);

                // 5. [Evaluate Model]
                var predictions = model.Transform(testData);
                var metrics = binaryClassificationContext.Evaluate(predictions, row => row.Label, row => row.Prediction);

                var predictionFunction = model.AsDynamic.MakePredictionFunction<SentimentData, SentimentPrediction>(environment);

                var prediction = predictionFunction.Predict(new SentimentData() { Comment = "Insert comment" });

                Console.WriteLine("Predicted sentiment is: " + prediction.PredictedLabel);
                Console.WriteLine();
                Console.WriteLine("PredictionModel quality metrics evaluation");
                Console.WriteLine("------------------------------------------");
                Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
                Console.WriteLine($"AUC: {metrics.Auc:P2}");
                Console.WriteLine($"Positive Precision: {metrics.PositivePrecision:P2}");
                Console.WriteLine($"Negative Precision: {metrics.NegativePrecision:P2}");
                Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:P2}");
                Console.WriteLine($"Negative Recall: {metrics.NegativeRecall:P2}");
                Console.WriteLine($"F1Score: {metrics.F1Score:P2}");

                Console.ReadLine();
            }           
        }
    }
}
