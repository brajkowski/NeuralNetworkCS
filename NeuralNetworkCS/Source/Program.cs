using System;
using System.Collections.Generic;

namespace NeuralNetworkCS
{
    class Program
    {
        static void Main()
        {
            var mnistData = new MnistData();
            mnistData.LoadAll();
            var sizes = new List<int> { 784, 30, 10 };
            var net = new Network(sizes,Activation.Sigmoid);
            net.SGD(ref mnistData,30,10,3f,true);
            net.SaveNetwork();
            Console.WriteLine("End of Main. Press any key...");
            Console.ReadKey();
        }
    }
}
