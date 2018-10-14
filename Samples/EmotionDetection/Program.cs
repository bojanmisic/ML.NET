using System;
using System.IO;
using System.Linq;
using EmotionDetection.DataModel;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Transforms;

namespace EmotionDetection
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var onnxModel = Path.Combine("Assets", "Model", "model.onnx");
            var imagePathsData = Path.Combine("Assets", "Data", "images.tsv");
            var imageFolder = Path.Combine("Assets", "Images");

            using (var environment = new ConsoleEnvironment())
            {
                var imageHeight = 64;
                var imageWidth = 64;

                var loader = TextLoader.CreateReader(environment, context => (
                    ImagePath: context.LoadText(0),
                    Name: context.LoadText(1)),
                    separator: '\t',
                    hasHeader: false);

                var data = loader.Read(new MultiFileSource(imagePathsData));

                var estimator = loader.MakeNewEstimator()
                    .Append(row => (
                        Name: row.Name,
                        input: row.ImagePath.LoadAsImage(imageFolder).AsGrayscale().Resize(imageWidth, imageHeight).ExtractPixels()))
                    .Append(row => (row.Name, EmotionScores: row.input.ApplyOnnxModel(onnxModel)));

                var model = estimator.Fit(data);

                var predictionFunction = model.AsDynamic.MakePredictionFunction<EmotionData, EmotionPrediction>(environment);

                var prediction = predictionFunction.Predict(new EmotionData() { ImagePath = "1.jpg" });

                int emotion = GetEmotion(prediction.PredictedLabels);

                Console.WriteLine(GetEmotionString(emotion));

                Console.ReadLine();
            }
        }

        private static string GetEmotionString(int emotion)
        {
            switch (emotion)
            {
                case 0: return "Neutral";
                case 1: return "Happiness";
                case 2: return "Surprise";
                case 3: return "Sadness";
                case 4: return "Anger";
                case 5: return "Disgust";
                case 6: return "Fear";
                case 7: return "Contemption";
                default: return "None";
            }
        }

        private static int GetEmotion(float[] predictedLabels)
        {
            var softmax = Softmax(Array.ConvertAll(predictedLabels, x => (double)x));
            return Array.IndexOf(softmax, softmax.Max());
        }

        private static float[] Softmax(double[] vector)
        {
            var zExp = vector.Select(Math.Exp);
            var sumZExp = zExp.Sum();
            return zExp.Select(i => (float)(i / sumZExp)).ToArray();
        }
    }
}
