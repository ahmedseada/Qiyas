// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A task's stores of output rows where the type's storage units share their scales (the block formats): a block stored in
// parts is re-encoded with the values already in it, which before its last part are whatever the output held, so a row
// that ends inside a block would leave the block's scale to those values. The rows' values are gathered in lanes until
// they fill whole units, and each unit is stored once, whole.
using System.Buffers;
using Adad.Abstraction;

namespace Adad.Cpu;

/// <summary>A task's row stores into an output of TN, in order: each row as it comes where the type's units do not share
/// scales or the rows are whole units; otherwise gathered and stored by whole units (the last one at Flush). A task whose
/// rows start on a unit boundary (RowChunk) stores every unit of its own whole; a run that does not (rows out of order, a
/// task starting inside a unit) is stored as it comes, as before.</summary>
internal struct UnitStore<TN, TLane> where TN : struct, IAdad<TN, TLane> where TLane : unmanaged
{
    private readonly byte[] _y;
    private readonly bool _gather;
    private TLane[]? _held;
    private long _at;                                        // the output index of _held[0], on a unit boundary
    private int _count;

    /// <summary>A task's store of rows of `cols` values: gathered where the type is a block format and a row ends inside
    /// its units.</summary>
    public UnitStore(byte[] y, int cols) : this(y, cols % TN.Granularity != 0) { }

    /// <summary>A store gathered where the type is a block format and `apart` (the caller's pieces end inside its units).</summary>
    public UnitStore(byte[] y, bool apart)
    {
        _y = y;
        _gather = apart && TN.IsFloat && TN.Granularity > 1;
        _at = -1;
    }

    /// <summary>The row's values at output index `at`.</summary>
    public void Put(ReadOnlySpan<TLane> row, long at)
    {
        if (!_gather) { TN.Store(row, _y, (int)at); return; }
        int g = TN.Granularity;
        if (_at < 0 || at != _at + _count)
        {
            Flush();
            if (at % g != 0) { TN.Store(row, _y, (int)at); return; }
            _at = at;
        }

        if (_held is null || _held.Length < _count + row.Length)
        {
            var bigger = ArrayPool<TLane>.Shared.Rent(Math.Max(_count + row.Length, 2 * g));
            if (_held is not null) { _held.AsSpan(0, _count).CopyTo(bigger); ArrayPool<TLane>.Shared.Return(_held); }
            _held = bigger;
        }

        row.CopyTo(_held.AsSpan(_count));
        _count += row.Length;
        int whole = _count / g * g;
        if (whole == 0) return;
        TN.Store(_held.AsSpan(0, whole), _y, (int)_at);
        _held.AsSpan(whole, _count - whole).CopyTo(_held);
        _at += whole;
        _count -= whole;
    }

    /// <summary>The values still held (a unit the task's rows end in: the output's last, or one the task ends on).</summary>
    public void Flush()
    {
        if (_count > 0) TN.Store(_held.AsSpan(0, _count), _y, (int)_at);
        _at = -1;
        _count = 0;
    }

    /// <summary>Flush, and the lanes given back.</summary>
    public void Done()
    {
        Flush();
        if (_held is not null) { ArrayPool<TLane>.Shared.Return(_held); _held = null; }
    }
}
