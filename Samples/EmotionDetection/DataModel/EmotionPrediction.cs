using Microsoft.ML.Runtime.Api;

namespace EmotionDetection.DataModel
{
    public class EmotionPrediction
    {
        [ColumnName("EmotionScores")]
        public float[] PredictedLabels;
    }
}