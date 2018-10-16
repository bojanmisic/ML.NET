namespace GithubIssueLabeler.DataModel
{
    using Microsoft.ML.Runtime.Api;

    public class GitHubIssuePrediction
    {
        [ColumnName("PredictedLabel")]
        public string Area;
    }
}