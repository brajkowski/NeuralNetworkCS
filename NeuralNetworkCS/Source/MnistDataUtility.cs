using System;
using System.IO;

namespace NeuralNetworkCS
{
    public static class MnistDataUtility
    {
        /// This method adopted from https://jamesmccaffrey.wordpress.com/2013/11/23/reading-the-mnist-data-set-with-c/
        /// and https://stackoverflow.com/questions/20967088/what-did-i-do-wrong-with-binaryreader-in-c.
        public static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian)
                Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }
    }
}
