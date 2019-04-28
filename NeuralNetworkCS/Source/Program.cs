using System;
using System.Collections.Generic;

namespace NeuralNetworkCS
{
    class Program
    {
        static void Main()
        {
            List<StockDataPoint> dataPoints = StockDataUtility.ReadStockFile(@"aa.us.csv");
            StockDataSet trainingSet = new StockDataSet(dataPoints);
            var sizes = new List<int> { 5, 30, 1 };
            var net = new Network(sizes, Activation.Sigmoid);
            //net.LoadNetwork();
            net.SGD(ref trainingSet, 3, 10, 3f, true);
            //net.SaveNetwork();
            Console.ReadKey();
        }
    }
}
