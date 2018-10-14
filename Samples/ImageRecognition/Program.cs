using System;
using System.IO;
using System.Linq;
using ImageRecognition.DataModel;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Transforms;

namespace ImageRecognition
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var tensorflowModel = Path.Combine("Assets", "Model", "tensorflow_inception_graph.pb");
            var tensorflowLabels = Path.Combine("Assets", "Model", "imagenet.tsv");
            var imagePathsData = Path.Combine("Assets", "Data", "images.tsv");
            var imageFolder = Path.Combine("Assets", "Images");

            using (var environment = new ConsoleEnvironment())
            {
                var imageHeight = 224;
                var imageWidth = 224;

                var loader = TextLoader.CreateReader(environment, context => (
                    ImagePath: context.LoadText(0),
                    Name: context.LoadText(1)),
                    separator: '\t',
                    hasHeader: false);

                var data = loader.Read(new MultiFileSource(imagePathsData));

                var estimator = loader.MakeNewEstimator()
                    .Append(row => (
                        Name: row.Name,
                        input: row.ImagePath.LoadAsImage(imageFolder).Resize(imageWidth, imageHeight).ExtractPixels(interleaveArgb: true, useAlpha: false, scale: 1, offset: 117)))
                    .Append(row => (row.Name, output2: row.input.ApplyTensorFlowGraph(tensorflowModel)));

                var model = estimator.Fit(data);

                var predictionFunction = model.AsDynamic.MakePredictionFunction<InceptionData, InceptionPrediction>(environment);

                var prediction = predictionFunction.Predict(new InceptionData() {ImagePath = "banana.jpg"});
                float maxValue = prediction.PredictedLabels.Max();
                int maxIndex = prediction.PredictedLabels.ToList().IndexOf(maxValue);

                string[] labels = File.ReadAllLines(tensorflowLabels);
                Console.WriteLine("Predicted label is: {0}", labels[maxIndex].Split('\t')[1]);

                Console.ReadLine();
            }
        }
    }
}
