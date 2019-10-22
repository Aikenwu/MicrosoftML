using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace BinaryClassification_DailyReport.DataStructures
{
    public class DailyReportPrediction
    {
        //预测值（是否合格）
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        //或然率（结果分布概率）
        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
