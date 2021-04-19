using System;
using System.Runtime.InteropServices;
using System.Text;

namespace OCRTest
{
    class Program
    {
        //[DllImport("../x64/OCRlib.dll")]
        [DllImport("C:/repo/OCR/x64/Debug/OCRlib.dll")]
        private extern static int PredictImgValue(char[] b0,char[] b1,char[] w0,char[] vs,char[] imgfile);
        [DllImport("C:/repo/OCR/x64/Debug/OCRlib.dll")]
        private extern static int OCRadd(int a, int b);
        static int Main(string[] args)
        {
            char[] b0 = "C:/repo/OCR/OCR/b0".ToCharArray();
            char[] b1 = "C:/repo/OCR/OCR/b1".ToCharArray();
            char[] w0 = "C:/repo/OCR/OCR/w0".ToCharArray();
            char[] w1 = "C:/repo/OCR/OCR/w1".ToCharArray();
            char[] imgfile = "d:/outputimg.jpg".ToCharArray();

            int value = PredictImgValue(b0,b1,w0,w1,imgfile);

            Console.WriteLine(value);
            return value;
        }
    }
}
