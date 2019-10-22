using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace BinaryClassification_DailyReport.DataStructures
{
    public class DailyReportData
    {
        [LoadColumn(0)]
        public bool Label { get; set; }
        [LoadColumn(1)]
        public string Content { get; set; }
    }
}
