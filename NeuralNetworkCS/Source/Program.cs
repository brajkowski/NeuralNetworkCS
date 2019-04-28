using System;
using System.Collections.Generic;

namespace NeuralNetworkCS
{
    class Program
    {
        static void Main()
        {
            List<StockDataPoint> dataPoints = StockDataUtility.ReadStockFile(@"aa.us.csv");
            StockDataSet trainingSet = new StockDataSet(dataPoints, true);
            int testingSize = (int)(trainingSet.getSize() * 0.20f);
            List<StockDataPoint> testingPoints = new List<StockDataPoint>();
            for (int i = 0; i < testingSize; i++)
            {
                testingPoints.Add(trainingSet.RandomRemoveFromSet());
            }
            StockDataSet testingSet = new StockDataSet(testingPoints, false);
            
            var sizes = new List<int> { 5, 30, 30, 1 };
            var net = new Network(sizes, Activation.Sigmoid);
            net.LoadNetwork(@"aaBest.dat");
            //net.SGD(ref trainingSet, ref testingSet, 50, 10, 0.1f, true);
            net.StockTest(ref testingSet);
            //net.SaveNetwork();
            Console.ReadKey();
        }
    }
}
