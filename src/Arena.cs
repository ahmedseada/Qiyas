// The device's buffer arena: inside a scope every Rent bumps a pointer in one region reserved from the shapes of earlier
// calls, and Return releases from the top (a buffer returned below the top is released when everything above it is), so a
// call's buffers are a stack in one allocation with no pool lookup and no driver call. A rent the region cannot hold comes
// from the pool for this call, and the scope's end grows the region to the peak the call would have needed had every rent
// been bumped — never inside a call — so the next call of the same shapes bumps everything. The pool stays as it was for
// rents outside any scope.
namespace Qiyas;

public sealed unsafe partial class Gpu
{
    /// <summary>Per-call counts of where buffers came from (cumulative; a gate or a report takes differences).</summary>
    public readonly record struct BufferCounts(long ArenaRents, long PoolRents, long DriverAllocations, long Overflows, long Leaks, long Strays);

    private ulong _region;
    private long _capacity, _top, _virtualTop, _need;
    private int _depth;
    private long _arenaRents, _poolRents, _driverAllocations, _overflows, _leaks, _strays;
    private long _scopeHighWater, _scopePeak;

    // The scope's buffers in rent order: those in the region and those from the pool (Pooled), freed or not.
    private readonly List<(ulong Pointer, long Size, bool Pooled, bool Freed)> _stack = [];

    public BufferCounts Counts => new(_arenaRents, _poolRents, _driverAllocations, _overflows, _leaks, _strays);

    /// <summary>The bytes the arena's region holds now.</summary>
    public long ArenaCapacity => _capacity;

    /// <summary>The last outermost scope's region high-water mark and the peak of its buffers alive at once (counting a
    /// buffer returned below the top until the top is released): equal when every rent of the scope fit the region.</summary>
    public (long HighWater, long Peak) LastScope => (_scopeHighWater, _scopePeak);

    /// <summary>A scope: until it is disposed, buffers come from the arena. Scopes nest; the outermost one's end resets the
    /// bump and grows the region when a call needed more.</summary>
    public ArenaScope Scope()
    {
        if (_depth == 0) { _scopeHighWater = 0; _scopePeak = 0; }
        _depth++;
        return new ArenaScope(this);
    }

    public readonly struct ArenaScope(Gpu gpu) : IDisposable
    {
        public void Dispose() => gpu.EndScope();
    }

    private void EndScope()
    {
        if (--_depth > 0) return;
        // Whatever the scope left unreturned is not held across the reset: counted (a gate requires none).
        foreach (var e in _stack) if (!e.Freed) _leaks++;
        _stack.Clear();
        _top = 0; _virtualTop = 0;
        if (_need > _capacity)
        {
            // Grown between calls: the work queued on the old region finishes first.
            Synchronize();
            if (_region != 0) Driver.Check(Driver.cuMemFree(_region), "cuMemFree");
            long bytes = (_need + (1 << 20) - 1) >> 20 << 20;
            Driver.Check(Driver.cuMemAlloc(out _region, (nuint)bytes), "cuMemAlloc");
            _driverAllocations++;
            _capacity = bytes;
        }
    }

    private ulong ArenaRent(long bytes)
    {
        long size = Math.Max(256, (bytes + 255) / 256 * 256);
        _arenaRents++;
        _virtualTop += size;
        _need = Math.Max(_need, _virtualTop);
        _scopePeak = Math.Max(_scopePeak, _virtualTop);
        if (_top + size <= _capacity)
        {
            ulong p = _region + (ulong)_top;
            _top += size;
            _scopeHighWater = Math.Max(_scopeHighWater, _top);
            _stack.Add((p, size, false, false));
            return p;
        }

        _overflows++;
        ulong pooled = PoolRent(bytes);
        _stack.Add((pooled, size, true, false));
        return pooled;
    }

    // True when the pointer was the arena's (released here); false sends it to the pool.
    private bool ArenaReturn(ulong pointer)
    {
        if (_depth == 0 && _stack.Count == 0) return InRegion(pointer) && Stray();
        for (int i = _stack.Count - 1; i >= 0; i--)
        {
            var e = _stack[i];
            if (e.Pointer != pointer || e.Freed) continue;
            _stack[i] = e with { Freed = true };
            if (e.Pooled) PoolReturn(pointer);
            // Release from the top while the top is freed.
            while (_stack.Count > 0 && _stack[^1].Freed)
            {
                var top = _stack[^1];
                _stack.RemoveAt(_stack.Count - 1);
                _virtualTop -= top.Size;
                if (!top.Pooled) _top -= top.Size;
            }

            return true;
        }

        return InRegion(pointer) && Stray();
    }

    private bool InRegion(ulong pointer) => _region != 0 && pointer >= _region && pointer < _region + (ulong)_capacity;

    // A region pointer returned after its scope ended: nothing to release (counted; a gate requires none).
    private bool Stray()
    {
        _strays++;
        return true;
    }
}
