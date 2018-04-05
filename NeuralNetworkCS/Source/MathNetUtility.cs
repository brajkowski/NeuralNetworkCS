using System;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworkCS
{
    public enum Activation { Sigmoid = 0, Relu };
    public static class MathNetUtility
    {
        public static float Relu(double num)
        {
            if (num >= 0d)
                return (float)(num);
            return 0f;
        }

        public static float ReluPrime(double num)
        {
            if (num >= 0d)
                return 1f;
            return 0f;
        }

        public static float Sigmoid(double num)
        {
            return (float)(1d / (1d + Math.Exp(-num)));
        }

        public static float SigmoidPrime(double num)
        {
            return (float)(Sigmoid(num) * (1d - Sigmoid(num)));
        }

        public static Vector<float> PointwiseActivation(this Vector<float> v, Activation a)
        {
            var temp = Vector<float>.Build.DenseOfVector(v);
            for (int i = 0; i < v.Count; i++)
            {
                switch(a)
                {
                    case Activation.Sigmoid:
                        temp[i] = Sigmoid((temp[i]));
                        break;
                    case Activation.Relu:
                        temp[i] = Relu((temp[i]));
                        break;
                }
            }
            return temp;
        }

        public static Vector<float> PointwisePrimeActivation(this Vector<float> v, Activation a)
        {
            var temp = Vector<float>.Build.DenseOfVector(v);
            for (int i = 0; i < v.Count; i++)
            {
                switch (a)
                {
                    case Activation.Sigmoid:
                        temp[i] = SigmoidPrime((temp[i]));
                        break;
                    case Activation.Relu:
                        temp[i] = ReluPrime((temp[i]));
                        break;
                }
            }
            return temp;
        }
    }
}
