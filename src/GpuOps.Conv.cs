// The device's convolution (Conv.cs): the same composition on device buffers.
namespace Qiyas;

public sealed partial class GpuOps
{
    /// <summary>y = conv(x, w); returns the product's lane.</summary>
    public string Conv<TN>(ConvArgs a, ulong x, ulong w, ulong y) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        ulong cols = gpu.Rent(f.Words((long)a.Rows * a.Cols) * (f.WordBits / 8));
        try
        {
            Layout<TN>(new LayoutArgs { Name = "im2col", X = [], P = a.P, Output = [], Count = a.Rows * a.Cols, InputCount = (int)a.XValues }, x, 0, cols);
            return MatMulView<TN>(new ProductView(a.OC, a.Cols, a.Rows, false, false), w, cols, y);
        }
        finally
        {
            gpu.Return(cols);
        }
    }

    /// <summary>dx and dw from dy, in the type's float lane storage; returns the lane's name.</summary>
    public string ConvBackward<TN>(ConvArgs a, ulong x, ulong w, ulong dy, ulong dx, ulong dw) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        bool own = f.Is("float32") || f.Is("float64");
        var rented = new List<ulong>();
        try
        {
            ulong lx = ToLane<TN>(x, a.XValues, rented), lw = ToLane<TN>(w, a.WValues, rented), ldy = ToLane<TN>(dy, a.YValues, rented);
            ulong ldx = own ? dx : gpu.Rent(a.XValues * lane.Size), ldw = own ? dw : gpu.Rent(a.WValues * lane.Size);
            if (!own) { rented.Add(ldx); rented.Add(ldw); }
            if (lane.Wide) ConvBackwardIn<Float64>(a, lx, lw, ldy, ldx, ldw); else ConvBackwardIn<Float32>(a, lx, lw, ldy, ldx, ldw);
            FromLane<TN>(ldx, dx, a.XValues);
            FromLane<TN>(ldw, dw, a.WValues);
            return lane.Name;
        }
        finally
        {
            foreach (var p in rented) gpu.Return(p);
        }
    }

    private void ConvBackwardIn<TS>(ConvArgs a, ulong x, ulong w, ulong dy, ulong dx, ulong dw) where TS : struct, IQiyasFloat<TS>
    {
        var f = Fmt.Of<TS>();
        long colBytes = f.Words((long)a.Rows * a.Cols) * (f.WordBits / 8);
        ulong cols = gpu.Rent(colBytes), dcols = gpu.Rent(colBytes);
        try
        {
            Layout<TS>(new LayoutArgs { Name = "im2col", X = [], P = a.P, Output = [], Count = a.Rows * a.Cols, InputCount = (int)a.XValues }, x, 0, cols);
            MatMulView<TS>(new ProductView(a.Rows, a.Cols, a.OC, true, false), w, dy, dcols);
            Layout<TS>(new LayoutArgs { Name = "col2im", X = [], P = a.P, Output = [], Count = (int)a.XValues, InputCount = a.Rows * a.Cols }, dcols, 0, dx);
            MatMulView<TS>(new ProductView(a.OC, a.Rows, a.Cols, false, true), dy, cols, dw);
        }
        finally
        {
            gpu.Return(cols); gpu.Return(dcols);
        }
    }
}
