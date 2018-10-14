using Microsoft.ML;
using Microsoft.ML.Runtime.Data;

namespace BikeSharing
{
    using Microsoft.ML.Data;
    using Microsoft.ML.StaticPipe;
    using Microsoft.ML.Trainers;
    using Microsoft.ML.Transforms;

    internal class Program
    {
        static void Main(string[] args)
        {
            var environment = new LocalEnvironment();
            var regressionContext = new RegressionContext(environment);

            //var multi = new MulticlassClassificationContext();

            //var aaa = new BinaryClassificationContext();

            //var bbb = new ClusteringContext();

            //bbb.Trainers.

            //aaa.Trainers.

            //regressionContext.Trainers.

            var reader = TextLoader.CreateReader(
                environment,
                context => (
                    Season: context.LoadFloat(2), 
                    Year: context.LoadFloat(3),
                    Month: context.LoadFloat(4),
                    Hour: context.LoadFloat(5),
                    Holiday: context.LoadText(6),
                    Weekday: context.LoadBool(7),
                    WorkingDay: context.LoadFloat(8),
                    Weather: context.LoadFloat(9),
                    Temperature: context.LoadFloat(10),
                    NormalizedTemperature: context.LoadFloat(11),
                    Humidity: context.LoadFloat(12),
                    Windspeed: context.LoadFloat(13),
                    Count: context.LoadFloat(16)
                ), 
                separator: ',', 
                hasHeader: true);

            var trainData = reader.Read(new MultiFileSource(@"Data/hour_train.csv"));
            var testData = reader.Read(new MultiFileSource(@"Data/hour_test.csv"));

            var estimator = reader.MakeNewEstimator();

            //estimator.Append(
            //    r => (
            //        Features: r.Season.ConcatWith(r.Month, r.Hour),
            //        Holiday: r.Weekday.
            //    ));
        }
    }
}
