using BinaryClassification_DailyReport.DataStructures;
using Microsoft.ML;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;

namespace BinaryClassification_DailyReport
{
    class Program
    {
        private static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "Nick_Report_learn_data.tsv");
        private static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "Nick_Report_test_data.tsv");
        private static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "MLModels", "Model.zip");

        static void Main(string[] args)
        {
            //创建要跨模型创建工作流对象共享的mlcontext
            //为跨多个训练的可重复/确定性结果设置随机种子。
            var mlContext = new MLContext(seed: 0);

            // STEP 1: 定型模型的训练数据集，评估模型的测试数据集
            IDataView dataTrainView = mlContext.Data.LoadFromTextFile<DailyReportData>(_trainDataPath, hasHeader: true);

            IDataView dataTestView = mlContext.Data.LoadFromTextFile<DailyReportData>(_testDataPath, hasHeader: true);

            // STEP 2:  数据特征化（按照管道所需的格式转换数据）          
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(DailyReportData.Content));

            // STEP 3:  根据学习算法添加学习管道                            
            var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // STEP 4: 根据定型模型的训练数据得到模型
            ITransformer trainedModel = trainingPipeline.Fit(dataTrainView);

            // STEP 5: 用评估模型的测试数据集评估模型的准确性
            Console.WriteLine();
            Console.WriteLine("=============== 评估模型的准确性，开始 ================");

            var predictions = trainedModel.Transform(dataTestView);
            var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");

            Console.WriteLine($"模型质量量度评估结果精度: {metrics.Accuracy:P2}");

            Console.WriteLine("=============== 评估模型的准确性，结束 ================");
            Console.WriteLine();


            // STEP 6: 评测完成之后保存定型的模型
            mlContext.Model.Save(trainedModel, dataTrainView.Schema, _modelPath);

            Console.WriteLine("模型->保存路径 {0}", _modelPath);

            // 创建与加载的训练模型相关的预测引擎
            var predEngine = mlContext.Model.CreatePredictionEngine<DailyReportData, DailyReportPrediction>(trainedModel);

            // TRY IT: 测试数据预测
            DailyReportData sampleStatement1 = new DailyReportData { Content = "开发了添加多个item到wishlist的功能" };
            DailyReportData sampleStatement2 = new DailyReportData { Content = "今天没有工作内容，打酱油" };

            // Score
            var resultprediction1 = predEngine.Predict(sampleStatement1);
            var resultprediction2 = predEngine.Predict(sampleStatement2);

            Console.WriteLine();
            Console.WriteLine("=============== 测试数据预测，开始 ===============");
            Console.WriteLine($"日报内容: {sampleStatement1.Content} | 预测结果: {(Convert.ToBoolean(resultprediction1.Prediction) ? "合格" : "不合格")} | 合格的可能性: {resultprediction1.Probability} ");
            Console.WriteLine();
            Console.WriteLine($"日报内容: {sampleStatement2.Content} | 预测结果: {(Convert.ToBoolean(resultprediction2.Prediction) ? "合格" : "不合格")} | 合格的可能性: {resultprediction2.Probability} ");
            Console.WriteLine("=============== 测试数据预测，结束 ================");

            Console.ReadLine();
        }
    }
}
