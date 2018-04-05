using System;
using System.IO;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworkCS
{
    public class MnistData
    {
        public Matrix<float> TrainImages;
        public Matrix<float> TestImages;
        public Vector<float> TrainLabels;
        public Vector<float> TestLabels;
        public enum SetType { Train = 0, Test = 1 };

        public MnistData()
        {
            // No initialization needed.
        }

        /// This method adopted from https://jamesmccaffrey.wordpress.com/2013/11/23/reading-the-mnist-data-set-with-c/.
        public void LoadImages(SetType type)
        {
            Console.WriteLine("Loading {0} image data...", type);

            string fileLocation = "Null";
            if (type == SetType.Train)
                fileLocation = @"MnistData/train-images.idx3-ubyte";
            else if (type == SetType.Test)
                fileLocation = @"MnistData/t10k-images.idx3-ubyte";

            var ifsImages = new FileStream(fileLocation, FileMode.Open);
            var brImages = new BinaryReader(ifsImages);

            // Read image file header bytes.
            int magic = brImages.ReadBigInt32();
            int numImages = brImages.ReadBigInt32();
            int numRows = brImages.ReadBigInt32();
            int numCols = brImages.ReadBigInt32();

            // Read in pixel data.
            float[] pixels = new float[numImages * numCols * numRows];
            for (int i = 0; i < numImages * numCols * numRows; i++)
                pixels[i] = brImages.ReadByte();

            // Store pixel data to class.
            if (type == SetType.Train)
            {
                TrainImages = Matrix<float>.Build.Dense(numRows * numCols, numImages, pixels);
                TrainImages = TrainImages.Divide(255f); // Normalize pixel data to be between 0 and 1.
            }
            else if (type == SetType.Test)
            {
                TestImages = Matrix<float>.Build.Dense(numRows * numCols, numImages, pixels);
                TestImages = TestImages.Divide(255f);   // Normalize pixel data to be between 0 and 1.
            }
            ifsImages.Close();
            Console.WriteLine("Done.");
        }

        /// This method adopted from https://jamesmccaffrey.wordpress.com/2013/11/23/reading-the-mnist-data-set-with-c/.
        public void LoadLabels(SetType type)
        {
            Console.WriteLine("Loading {0} label data...", type);

            string fileLocation = "Null";
            if (type == SetType.Train)
                fileLocation = @"MnistData/train-labels.idx1-ubyte";
            else if (type == SetType.Test)
                fileLocation = @"MnistData/t10k-labels.idx1-ubyte";

            var ifsLabels = new FileStream(fileLocation, FileMode.Open);
            var brLabels = new BinaryReader(ifsLabels);

            // Read label file header bytes.
            int magic = brLabels.ReadBigInt32();
            int numLabels = brLabels.ReadBigInt32();

            // Read in label data.
            float[] labels = new float[numLabels];
            for (int i = 0; i < numLabels; i++)
            {
                labels[i] = brLabels.ReadByte();
            }

            // Store label data to class.
            if (type == SetType.Train)
                TrainLabels = Vector<float>.Build.Dense(labels);
            else if (type == SetType.Test)
                TestLabels = Vector<float>.Build.Dense(labels);
            ifsLabels.Close();
            Console.WriteLine("Done.");
        }

        public void LoadAll()
        {
            LoadImages(SetType.Test);
            LoadLabels(SetType.Test);
            LoadImages(SetType.Train);
            LoadLabels(SetType.Train);
        }
    }
}
