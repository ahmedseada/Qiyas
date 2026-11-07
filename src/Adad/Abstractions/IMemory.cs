// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Adad;

/// <summary>A device's memory: buffers of stored values (TBuffer) and of indices (TIndex, 32-bit), rented, filled from the host
/// and read back. Operations are ordered on the device; Synchronize waits for them.</summary>
public interface IMemory<TBuffer, TIndex>
{
    /// <summary>A buffer of at least <paramref name="bytes"/> bytes.</summary>
    TBuffer Rent(long bytes);

    void Return(TBuffer buffer);

    /// <summary>The host's bytes into the buffer, from its start.</summary>
    void Upload(ReadOnlySpan<byte> source, TBuffer destination);

    /// <summary>The buffer's first destination.Length bytes to the host.</summary>
    void Download(TBuffer source, Span<byte> destination);

    void Copy(TBuffer source, TBuffer destination, long bytes);

    /// <summary>A buffer of at least <paramref name="count"/> indices.</summary>
    TIndex RentIndices(int count);

    void ReturnIndices(TIndex indices);

    void UploadIndices(ReadOnlySpan<int> source, TIndex destination);

    void DownloadIndices(TIndex source, Span<int> destination);

    /// <summary>Waits for the operations issued so far.</summary>
    void Synchronize();
}
