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
            //net.LoadNetwork();
            //Console.WriteLine(net.MnistTest(ref mnistData));
            //net.SGD(ref mnistData,30,10,3f,true);
            //net.SaveNetwork();
            //net.SaveNetworkCSV();
            /*
            for(int i = 0; i < 10; i++)
            {
                string path = "VisualizedImages/" + "d" + mnistData.TestLabels[i] + "_image" + i + ".png";
                mnistData.CreateImage(mnistData.TestImages.Column(i),path);
            }
            */
            mnistData.CreateCSV(mnistData.TestImages.Column(0), mnistData.TestLabels[0].ToString(), "mnist_image_0.csv");
            Console.WriteLine("End of Main. Press any key...");
            Console.ReadKey();
        }
    }
}
