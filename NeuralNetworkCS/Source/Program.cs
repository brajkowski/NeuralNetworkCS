using System;
using System.Collections.Generic;

namespace NeuralNetworkCS
{
    class Program
    {
        static void Main()
        {
            //var mnistData = new MnistData();
            //mnistData.LoadAll();
            //var sizes = new List<int> { 784, 30, 10 };
            //var net = new Network(sizes,Activation.Sigmoid);
            ////net.LoadNetwork();
            //net.SGD(ref mnistData,3,10,3f,true);
            //net.SaveNetwork();
            ////net.SaveNetworkCSV();
            //Console.WriteLine("End of Main. Press any key...");
            //Console.ReadKey();
            List<StockDataPoint> dataPoints = StockDataUtility.ReadStockFile(@"aa.us.csv");
            StockDataSet trainingSet = new StockDataSet(dataPoints);
            StockDataForNetwork networkData = trainingSet.GetNextNetworkData();
            while (networkData != null)
            {
                Console.WriteLine(networkData.OutputLayer);
                networkData = trainingSet.GetNextNetworkData();
            }
            //Console.WriteLine(dataPoints.Count);
            Console.ReadKey();
        }
    }
}
