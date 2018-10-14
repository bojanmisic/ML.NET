using Microsoft.ML.Runtime.Api;

namespace ImageRecognition.DataModel
{
    public class InceptionPrediction
    {
        [ColumnName("output2")]
        public float[] PredictedLabels;
    }
}