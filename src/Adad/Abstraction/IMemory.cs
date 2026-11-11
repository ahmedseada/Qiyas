// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Adad.Abstraction;

/// <summary>A device's memory: buffers of stored values (TBuffer) and of indices (TIndex, 32-bit), rented, filled from the host
/// and read back, whole or a range of bytes at an offset (a range lies within its buffer; a copy's two ranges do not
/// overlap). Operations are ordered on the device; Synchronize waits for them.</summary>
public interface IMemory<TBuffer, TIndex>
{
    /// <summary>A buffer of at least <paramref name="bytes"/> bytes.</summary>
    TBuffer Rent(long bytes);

    /// <summary>A buffer of at least <paramref name="bytes"/> bytes, each of them 0 (whatever the buffer held before).</summary>
    TBuffer RentZeroed(long bytes);

    void Return(TBuffer buffer);

    /// <summary>Zero in the buffer's <paramref name="bytes"/> bytes from byte <paramref name="offset"/>; no other byte changed.</summary>
    void Clear(TBuffer buffer, long offset, long bytes);

    /// <summary>The host's bytes into the buffer, from its start.</summary>
    void Upload(ReadOnlySpan<byte> source, TBuffer destination);

    /// <summary>The host's bytes into the buffer from byte <paramref name="offset"/>; no other byte changed.</summary>
    void Upload(ReadOnlySpan<byte> source, TBuffer destination, long offset);

    /// <summary>The buffer's first destination.Length bytes to the host.</summary>
    void Download(TBuffer source, Span<byte> destination);

    /// <summary>The buffer's destination.Length bytes from byte <paramref name="offset"/> to the host.</summary>
    void Download(TBuffer source, long offset, Span<byte> destination);

    void Copy(TBuffer source, TBuffer destination, long bytes);

    /// <summary><paramref name="bytes"/> bytes of the source from byte <paramref name="sourceOffset"/> into the destination
    /// from byte <paramref name="destinationOffset"/> (one buffer's two ranges when they do not overlap); no other byte
    /// changed.</summary>
    void Copy(TBuffer source, long sourceOffset, TBuffer destination, long destinationOffset, long bytes);

    /// <summary>A buffer of at least <paramref name="count"/> indices.</summary>
    TIndex RentIndices(int count);

    void ReturnIndices(TIndex indices);

    void UploadIndices(ReadOnlySpan<int> source, TIndex destination);

    void DownloadIndices(TIndex source, Span<int> destination);

    /// <summary>Waits for the operations issued so far.</summary>
    void Synchronize();
}
