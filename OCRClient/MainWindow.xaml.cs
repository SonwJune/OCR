using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Ink;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Runtime.InteropServices;

namespace OCRClient
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        [DllImport("C:/repo/OCR/x64/Debug/OCRlib.dll")]
        private extern static int PredictImgValue(char[] b0, char[] b1, char[] w0, char[] vs, char[] imgfile);
        [DllImport("C:/repo/OCR/x64/Debug/OCRlib.dll")]
        private extern static void ResizeImg(char[] filename);

        public MainWindow()
        {
            InitializeComponent();
            inkSig.DefaultDrawingAttributes.Height = 15;
            inkSig.DefaultDrawingAttributes.Width = 15;
            inkSig.DefaultDrawingAttributes.Color = Color.FromRgb(255, 255, 255);
        }

        private void btnSave_Click(object sender, RoutedEventArgs e)
        {
            
            string sigPath = "d:/outputimg.jpg";

            MemoryStream ms = new MemoryStream();
            FileStream fs = new FileStream(sigPath, FileMode.Create);

            RenderTargetBitmap rtb = new RenderTargetBitmap((int)inkSig.Width, (int)inkSig.Height, 96d, 96d, PixelFormats.Default);


            double width = inkSig.ActualWidth;
            double height = inkSig.ActualHeight;
            DrawingVisual dv = new DrawingVisual();
            using (DrawingContext dc = dv.RenderOpen())
            {
                VisualBrush vb = new VisualBrush(inkSig);
                dc.DrawRectangle(vb, null, new Rect(new System.Windows.Point(), new System.Windows.Size(width, height)));
            }
            rtb.Render(dv);



            //rtb.Render(inkSig);
            JpegBitmapEncoder encoder = new JpegBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(rtb));

            encoder.Save(fs);
            fs.Close();

            char[] imgfile = "d:/outputimg.jpg".ToCharArray();
            ResizeImg(imgfile);

            char[] b0 = "C:/repo/OCR/OCR/b0".ToCharArray();
            char[] b1 = "C:/repo/OCR/OCR/b1".ToCharArray();
            char[] w0 = "C:/repo/OCR/OCR/w0".ToCharArray();
            char[] w1 = "C:/repo/OCR/OCR/w1".ToCharArray();

            int value = PredictImgValue(b0, b1, w0, w1, imgfile);
            outlbl.Content = "预测值：" + value.ToString();
        }


        private void btnClear_Click(object sender, RoutedEventArgs e)
        {
            inkSig.Strokes.Clear();
        }
    }
}
