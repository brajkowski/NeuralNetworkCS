using System;
using System.IO;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworkCS
{
    class Network
    {
        private List<int> vSizes;   // Using List<int> since Math.NET doesnt support Vector<int>.
        private Vector<float> vY;
        private List<Vector<float>> mNeurons;
        private List<Vector<float>> mZ;
        private List<Vector<float>> mError;
        private List<Vector<float>> mBiases;
        private List<Vector<float>> mSumB;
        private List<Matrix<float>> mWeights;
        private List<Matrix<float>> mSumW;
        private int lastLayerIndex;
        private Activation activation;

        public double Accuracy { get; set; }

        public Network(List<int> sizes, Activation a)
        {
            Accuracy = 0d;
            vSizes = sizes;
            activation = a;
            lastLayerIndex = vSizes.Count - 1;

            vY = Vector<float>.Build.Dense(vSizes[lastLayerIndex], 0f);
            mNeurons = new List<Vector<float>>();
            mZ = new List<Vector<float>>();
            mError = new List<Vector<float>>();
            mBiases = new List<Vector<float>>();
            mSumB = new List<Vector<float>>();
            mWeights = new List<Matrix<float>>();
            mSumW = new List<Matrix<float>>();

            // Size first index of matrices for unique cases.
            mZ.Add(Vector<float>.Build.Dense(1, 0f));
            mError.Add(Vector<float>.Build.Dense(1, 0f));
            mBiases.Add(Vector<float>.Build.Dense(1, 0f));
            mSumB.Add(Vector<float>.Build.Dense(1, 0f));
            mNeurons.Add(Vector<float>.Build.Dense(vSizes[0], 0f));
            mWeights.Add(Matrix<float>.Build.Dense(1, 1, 0f));
            mSumW.Add(Matrix<float>.Build.Dense(1, 1, 0f));

            for (int l = 1; l < vSizes.Count; l++)
            {
                mZ.Add(Vector<float>.Build.Dense(vSizes[l], 0f));
                mError.Add(Vector<float>.Build.Dense(vSizes[l], 0f));
                mBiases.Add(Vector<float>.Build.Dense(vSizes[l], 0f));
                mSumB.Add(Vector<float>.Build.Dense(vSizes[l], 0f));
                mNeurons.Add(Vector<float>.Build.Dense(vSizes[l], 0f));
                mWeights.Add(Matrix<float>.Build.Dense(vSizes[l], vSizes[l - 1], 0f));
                mSumW.Add(Matrix<float>.Build.Dense(vSizes[l], vSizes[l - 1], 0f));
            }
        }

        public void SetInputLayer(Vector<float> v)
        {
            v.CopyTo(mNeurons[0]);
        }

        public void RandomizeParameters()
        {
            for (int l = 1; l < mWeights.Count; l++)
            {
                Matrix<float>.Build.Random(mWeights[l].RowCount, mWeights[l].ColumnCount).CopyTo(mWeights[l]);
                Vector<float>.Build.Random(mBiases[l].Count).CopyTo(mBiases[l]);
            }
        }

        public void FeedForward()
        {
            for (int l = 1; l < vSizes.Count; l++)
                mZ[l].Clear();

            for (int l = 1; l < vSizes.Count; l++)
            {
                mZ[l] = mWeights[l] * mNeurons[l - 1] + mBiases[l];
                mNeurons[l] = mZ[l].PointwiseActivation(activation);
            }
        }

        public void SetOutputLayer(int y)
        {
            // Convert integer label to output vector.
            for (int j = 0; j < vY.Count; j++)
            {
                if (j == y)
                    vY[j] = 1f;
                else
                    vY[j] = 0f;
            }
        }

        public void SetOutputLayer(Vector<float> v)
        {
            v.CopyTo(vY);
        }

        public void OutputError()
        {
            mError[lastLayerIndex] = mNeurons[lastLayerIndex].Subtract(vY);
            mError[lastLayerIndex] = mError[lastLayerIndex].PointwiseMultiply(mZ[lastLayerIndex].PointwisePrimeActivation(activation));
        }

        public void Backprop()
        {
            for (int l = lastLayerIndex - 1; l > 0; l--)
            {
                mError[l] = mWeights[l + 1].TransposeThisAndMultiply(mError[l + 1]);
                mError[l] = mError[l].PointwiseMultiply(mZ[l].PointwisePrimeActivation(activation));
            }
        }

        public void SGD(ref MnistData data, int epochs, int batchSize, float learningRate, bool output)
        {
            MathNet.Numerics.Control.UseSingleThread(); // Single thread is optimal over multithreading.

            Console.WriteLine("Training...");

            for (int l = 0; l < vSizes.Count; l++)
            {
                mSumW[l].Clear();
                mSumB[l].Clear();
            }

            RandomizeParameters();

            for (int i = 0; i < epochs; i++)
            {
                for (int j = 0; j < data.TrainImages.ColumnCount; j++)
                {
                    SetInputLayer(data.TrainImages.Column(j));
                    FeedForward();
                    SetOutputLayer((int)(data.TrainLabels[j]));
                    OutputError();
                    Backprop();

                    // Perform summation calculations over the training images.
                    for (int l = lastLayerIndex; l > 0; l--)
                    {
                        mSumW[l] += mError[l].ToColumnMatrix() * mNeurons[l - 1].ToRowMatrix();
                        mSumB[l] += mError[l];
                    }

                    // Update the weights and biases after the mini-batch has been processed.
                    if (j % batchSize == 0 && j > 0)
                    {
                        for (int l = 0; l < vSizes.Count; l++)
                        {
                            mWeights[l] += mSumW[l].Multiply(-learningRate / batchSize);
                            mBiases[l] += mSumB[l].Multiply(-learningRate / batchSize);
                            mSumW[l].Clear();
                            mSumB[l].Clear();
                        }
                    }
                }
                if (output == true) Console.WriteLine("Epoch {0}: {1} / {2}", i, MnistTest(ref data), data.TestLabels.Count);
            }
            Console.WriteLine("Training Complete.");
        }

        public void SGD(ref StockDataSet data, int epochs, int batchSize, float learningRate, bool output)
        {
            MathNet.Numerics.Control.UseSingleThread(); // Single thread is optimal over multithreading.

            Console.WriteLine("Training...");

            for (int l = 0; l < vSizes.Count; l++)
            {
                mSumW[l].Clear();
                mSumB[l].Clear();
            }

            RandomizeParameters();

            for (int i = 0; i < epochs; i++)
            {
                StockDataForNetwork networkData = data.GetNextNetworkData();
                int j = 0;
                while (networkData != null)
                {
                    SetInputLayer(networkData.InputLayer);
                    FeedForward();
                    SetOutputLayer(networkData.OutputLayer);
                    OutputError();
                    Backprop();

                    // Perform summation calculations over the training images.
                    for (int l = lastLayerIndex; l > 0; l--)
                    {
                        mSumW[l] += mError[l].ToColumnMatrix() * mNeurons[l - 1].ToRowMatrix();
                        mSumB[l] += mError[l];
                    }

                    // Update the weights and biases after the mini-batch has been processed.
                    if (j % batchSize == 0 && j > 0)
                    {
                        for (int l = 0; l < vSizes.Count; l++)
                        {
                            mWeights[l] += mSumW[l].Multiply(-learningRate / batchSize);
                            mBiases[l] += mSumB[l].Multiply(-learningRate / batchSize);
                            mSumW[l].Clear();
                            mSumB[l].Clear();
                        }
                    }
                    networkData = data.GetNextNetworkData();
                    j++;
                }
                if (output == true) Console.WriteLine("Epoch {0}: {1} / {2}", i, StockTest(ref data), data.getSize());
                data.Reset();
            }
            Console.WriteLine("Training Complete.");
        }

        public int MnistTest(ref MnistData data)
        {
            int correct = 0;
            int total = data.TestLabels.Count;
            for (int i = 0; i < total; i++)
            {
                SetInputLayer(data.TestImages.Column(i));
                FeedForward();
                if ((mNeurons[lastLayerIndex].MaximumIndex()) == data.TestLabels[i])
                {
                    correct++;
                }
            }
            Console.WriteLine(correct / total);
            this.Accuracy = (double)correct / (double)total;
            return correct;
        }

        public int StockTest(ref StockDataSet data)
        {
            data.Reset();
            int correct = 0;
            int total = data.getSize();
            for (int i = 0; i < total; i++)
            {
                StockDataForNetwork networkData = data.GetNextNetworkData();
                SetInputLayer(networkData.InputLayer);
                FeedForward();
                float predictedValue = mNeurons[lastLayerIndex][0];
                if (predictedValue >= 0.5f && networkData.OutputLayer[0] >= 0.5f)
                {
                    correct++;
                    continue;
                }
                if (predictedValue < 0.5f && networkData.OutputLayer[0] < 0.5f)
                {
                    correct++;
                    continue;
                }
            }
            Console.WriteLine(correct / total);
            this.Accuracy = (double)correct / (double)total;
            data.Reset();
            return correct;
        }

        /// <summary>
        /// Creates a binary output file of the network of the following structure:
        /// <para>Number of layers, neurons in each layer, weights, and then biases.</para>
        /// </summary>
        public void SaveNetwork()
        {
            var ofs = new FileStream(@"network2.dat", FileMode.OpenOrCreate);
            var bw = new BinaryWriter(ofs);
            bw.Write(vSizes.Count);
            for (int i = 0; i < vSizes.Count; i++)
            {
                bw.Write(vSizes[i]);
            }
            for (int l = 1; l < vSizes.Count; l++)
                for (int j = 0; j < vSizes[l]; j++)
                    for (int k = 0; k < vSizes[l - 1]; k++)
                        bw.Write(mWeights[l][j,k]);
            for (int l = 1; l < vSizes.Count; l++)
                for (int j = 0; j < vSizes[l]; j++)
                    bw.Write(mBiases[l][j]);
            ofs.Close();
        }

        /// <summary>
        /// Loads a binary output file of the network of the following structure:
        /// <para>Number of layers, neurons in each layer, weights, and then biases.</para>
        /// </summary>
        public void LoadNetwork()
        {
            var ifs = new FileStream(@"network.dat", FileMode.Open);
            var br = new BinaryReader(ifs);
            var layers = br.ReadInt32();
            var tempSizes = new List<int>();
            for (int i = 0; i < layers; i++)
                tempSizes.Add(br.ReadInt32());
            vSizes = tempSizes;
            for (int l = 1; l < vSizes.Count; l++)
                for (int j = 0; j < vSizes[l]; j++)
                    for (int k = 0; k < vSizes[l - 1]; k++)
                        mWeights[l][j, k] = br.ReadSingle();
            for (int l = 1; l < vSizes.Count; l++)
                for (int j = 0; j < vSizes[l]; j++)
                    mBiases[l][j] = br.ReadSingle();
            ifs.Close();
        }

        public void SaveNetworkCSV()
        {
            // Weights file.
            var ofs = new FileStream(@"network_weights.csv", FileMode.OpenOrCreate);
            var sw = new StreamWriter(ofs);
            var temp = "";
            var header = "From layer" + "," + "From neuron" + "," + "To layer" + "," + "To neuron" + "," + "Weight";
            sw.WriteLine(header);
            for (int l = 1; l < vSizes.Count; l++)
                for (int j = 0; j < vSizes[l]; j++)
                    for (int k = 0; k < vSizes[l - 1]; k++)
                    {
                        temp = (l - 1).ToString() + "," + (k).ToString() + "," + (l).ToString() + "," + (j).ToString() + "," + mWeights[l][j,k].ToString();
                        sw.WriteLine(temp);
                    }
            sw.Close();

            // Biases file.
            ofs = new FileStream(@"network_biases.csv", FileMode.OpenOrCreate);
            sw = new StreamWriter(ofs);
            temp = "";
            header = "Layer" + "," + "Neuron" + "," + "Bias";
            sw.WriteLine(header);
            for (int l = 1; l < vSizes.Count; l++)
                for (int j = 0; j < vSizes[l]; j++)
                {
                    temp = (l).ToString() + "," + (j).ToString() + "," + mBiases[l][j].ToString();
                    sw.WriteLine(temp);
                }
            sw.Close();

            // Statistics file.
            ofs = new FileStream(@"network_stats.csv", FileMode.OpenOrCreate);
            sw = new StreamWriter(ofs);
            temp = "";

            // Count all neurons.
            int neuronCount = 0;
            for (int l = 0; l < vSizes.Count; l++)
                neuronCount += mNeurons[l].Count;

            // Count hidden neurons.
            int hiddenNeuronCount = 0;
            for (int l = 1; l < vSizes.Count - 1; l++)
                hiddenNeuronCount += mNeurons[l].Count;

            // Count weights.
            int weightCount = 0;
            for (int l = 1; l < vSizes.Count; l++)
                weightCount += mNeurons[l - 1].Count * mNeurons[l].Count;

            // Count biases.
            int biasCount = 0;
            for (int l = 1; l < vSizes.Count; l++)
                biasCount += mNeurons[l].Count;

            temp = "Layout";
            for (int l = 0; l < vSizes.Count; l++)
            {
                temp += "," + vSizes[l];
            }
            sw.WriteLine(temp);
            sw.WriteLine("Total neurons" + "," + neuronCount);
            sw.WriteLine("Hidden neurons" + "," + hiddenNeuronCount);
            sw.WriteLine("Total weights" + "," + weightCount);
            sw.WriteLine("Total biases" + "," + biasCount);
            sw.WriteLine("Test accuracy" + "," + Accuracy);
            sw.Close();
        }
    }
}
