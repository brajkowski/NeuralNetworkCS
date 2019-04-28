using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualBasic.FileIO;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworkCS
{
    class StockDataForNetwork
    {
        public Vector<float> InputLayer;
        public Vector<float> OutputLayer;
    }

    class StockDataSet
    {
        private List<StockDataPoint> dataSet;
        private int index;
        private int sizeLimit;
        public StockDataSet(List<StockDataPoint> dataPoints, bool shouldNormalize)
        {
            index = 0;
            sizeLimit = dataPoints.Count - 1; // The last data point cannot be used to create a network data point.
            dataSet = dataPoints;
            if (shouldNormalize)
            {
                Normalize();
            }
        }

        public int getSize()
        {
            return sizeLimit;
        }

        public void Reset()
        {
            index = 0;
        }
        
        private void Normalize()
        {
            float lowOpen = float.MaxValue;
            float highOpen = float.MinValue;
            float lowHigh = float.MaxValue;
            float highHigh = float.MinValue;
            float lowLow = float.MaxValue;
            float highLow = float.MinValue;
            float lowClose = float.MaxValue;
            float highClose = float.MinValue;
            float lowVolume = float.MaxValue;
            float highVolume = float.MinValue;

            foreach(StockDataPoint p in dataSet)
            {
                if (p.Open < lowOpen) { lowOpen = p.Open; }
                if (p.Open > highOpen) { highOpen = p.Open; }
                if (p.High < lowHigh) { lowHigh = p.High; }
                if (p.High > highHigh) { highHigh = p.High; }
                if (p.Low < lowLow) { lowLow = p.Low; }
                if (p.Low > highLow) { highLow = p.Low; }
                if (p.Close < lowClose) { lowClose = p.Close; }
                if (p.Close > highClose) { highClose = p.Close; }
                if (p.Volume < lowVolume) { lowVolume = p.Volume; }
                if (p.Volume > highVolume) { highVolume = p.Volume; }
            }

            foreach(StockDataPoint p in dataSet)
            {
                p.Open = StockDataUtility.NormalizeValue(lowOpen, highOpen, p.Open);
                p.High = StockDataUtility.NormalizeValue(lowHigh, highHigh, p.High);
                p.Low = StockDataUtility.NormalizeValue(lowLow, highLow, p.Low);
                p.Close = StockDataUtility.NormalizeValue(lowClose, highClose, p.Close);
                p.Volume = StockDataUtility.NormalizeValue(lowVolume, highVolume, p.Volume);
            }
        }

        public StockDataForNetwork GetNextNetworkData()
        {
            if (index == sizeLimit)
            {
                return null;
            }
            var inputData = new float[5];
            inputData[0] = dataSet[index].Open;
            inputData[1] = dataSet[index].High;
            inputData[2] = dataSet[index].Low;
            inputData[3] = dataSet[index].Close;
            inputData[4] = dataSet[index].Volume;
            Vector<float> inputVector = Vector<float>.Build.Dense(inputData);

            var outputData = new float[1];
            if (dataSet[index + 1].Close >= dataSet[index].Close)
            {
                outputData[0] = 1f;
            }
            else
            {
                outputData[0] = 0f;
            }
            Vector<float> outputVector = Vector<float>.Build.Dense(outputData);
            var ret = new StockDataForNetwork();
            ret.InputLayer = inputVector;
            ret.OutputLayer = outputVector;
            index++;
            return ret;
        }

        public StockDataPoint RandomRemoveFromSet()
        {
            Random r = new Random();
            int indexToRemove = r.Next(0, sizeLimit);
            StockDataPoint ret = dataSet[indexToRemove];
            dataSet.RemoveAt(indexToRemove);
            sizeLimit--;
            return ret;
        }
    }

    class StockDataPoint
    {
        public float Open;
        public float High;
        public float Low;
        public float Close;
        public float Volume;

        public StockDataPoint(float open, float high , float low, float close, float volume)
        {
            Open = open;
            High = high;
            Low = low;
            Close = close;
            Volume = volume;
        }

        public override string ToString()
        {
            StringBuilder s = new StringBuilder();
            s.Append(Open);
            s.Append("\t");
            s.Append(High);
            s.Append("\t");
            s.Append(Low);
            s.Append("\t");
            s.Append(Close);
            s.Append("\t");
            s.Append(Volume);
            s.Append("\t");
            return s.ToString();
        }
    }

    //class StockTrainingPoint
    //{
    //    private Vector<float> inputColumn;
    //    private Vector<float> outputColumn;

    //    public StockTrainingPoint(StockDataPoint current, StockDataPoint next)
    //    {
    //        inputColumn = new Vector<float>();
    //        inputColumn.Add(current.)
    //    }
    //}

    class StockDataUtility
    {
        public static List<StockDataPoint> ReadStockFile(string path)
        {
            List<StockDataPoint> dataPoints = new List<StockDataPoint>();
            TextFieldParser csvParser = new TextFieldParser(path);
            csvParser.TextFieldType = FieldType.Delimited;
            csvParser.SetDelimiters(",");
            csvParser.ReadFields(); // Skip first line.
            while (!csvParser.EndOfData)
            {
                string[] p = csvParser.ReadFields();
                float open = float.Parse(p[1]);
                float high = float.Parse(p[2]);
                float low = float.Parse(p[3]);
                float close = float.Parse(p[4]);
                float volume = float.Parse(p[5]);
                StockDataPoint dataPoint = new StockDataPoint(open, high, low, close, volume);
                //Console.WriteLine(dataPoint.ToString());
                dataPoints.Add(dataPoint);

            }
            csvParser.Close();
            return dataPoints;
        }

        public static float NormalizeValue(float low, float high, float actual)
        {
            float num = actual - low;
            float den = high - low;
            return num / den;
        }
    }
}
