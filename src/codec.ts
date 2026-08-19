import { crc32 } from 'node:zlib';
import type { VectorInput } from './types.js';

/**
 * On-disk layout (see README):
 *   0   4      magic "EMBC"
 *   4   1      format version
 *   5   1      dtype (1 = float32)
 *   6   2      reserved
 *   8   4      dim, uint32 little-endian
 *   12  4      crc32 of the payload
 *   16  4*dim  payload, float32 little-endian
 */
const MAGIC = 'EMBC';
const FORMAT_VERSION = 1;
const DTYPE_FLOAT32 = 1;
const HEADER_SIZE = 16;

export interface DecodedVector {
  vector: Float32Array;
  dim: number;
}

/**
 * Narrow any accepted vector shape to Float32Array. This is also where a
 * number[] pays its precision-loss cost, once, instead of on every read.
 */
export function toFloat32(input: VectorInput): Float32Array {
  if (input instanceof Float32Array) return input;
  if (input instanceof Float64Array) return Float32Array.from(input);
  if (Array.isArray(input)) return Float32Array.from(input);
  throw new TypeError(`vector must be a Float32Array, Float64Array, or number[], got ${typeof input}`);
}

/**
 * Elements are written with an explicit little-endian DataView-style call
 * rather than a raw ArrayBuffer copy, so the file format does not depend on
 * the host's endianness.
 */
export function encodeVector(input: VectorInput): Buffer {
  const vector = toFloat32(input);
  const dim = vector.length;
  const buf = Buffer.alloc(HEADER_SIZE + dim * 4);

  buf.write(MAGIC, 0, 'ascii');
  buf.writeUInt8(FORMAT_VERSION, 4);
  buf.writeUInt8(DTYPE_FLOAT32, 5);
  buf.writeUInt32LE(dim, 8);

  const payload = buf.subarray(HEADER_SIZE);
  for (let i = 0; i < dim; i++) {
    payload.writeFloatLE(vector[i], i * 4);
  }

  buf.writeUInt32LE(crc32(payload) >>> 0, 12);
  return buf;
}

/**
 * Returns null rather than throwing on any structural or checksum failure,
 * so a caller reading from disk can treat a corrupt file the same way it
 * treats a missing one: discard and recompute.
 */
export function decodeVector(buf: Buffer | Uint8Array, expectedDim?: number | null): DecodedVector | null {
  if (buf.length < HEADER_SIZE) return null;
  const header = Buffer.isBuffer(buf) ? buf : Buffer.from(buf.buffer, buf.byteOffset, buf.byteLength);

  if (header.toString('ascii', 0, 4) !== MAGIC) return null;
  if (header.readUInt8(4) !== FORMAT_VERSION) return null;
  if (header.readUInt8(5) !== DTYPE_FLOAT32) return null;

  const dim = header.readUInt32LE(8);
  if (header.length !== HEADER_SIZE + dim * 4) return null;
  if (expectedDim != null && dim !== expectedDim) return null;

  const payload = header.subarray(HEADER_SIZE);
  const storedCrc = header.readUInt32LE(12);
  if ((crc32(payload) >>> 0) !== storedCrc) return null;

  const vector = new Float32Array(dim);
  for (let i = 0; i < dim; i++) {
    vector[i] = payload.readFloatLE(i * 4);
  }

  return { vector, dim };
}
