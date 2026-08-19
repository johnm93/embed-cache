import { test } from 'node:test';
import assert from 'node:assert/strict';
import { encodeVector, decodeVector, toFloat32 } from '../src/codec.ts';

test('round-trips a Float32Array', () => {
  const original = Float32Array.from([1, -2.5, 0, 3.140625, 1e10]);
  const decoded = decodeVector(encodeVector(original));
  assert.ok(decoded);
  assert.equal(decoded.dim, original.length);
  assert.deepEqual(decoded.vector, original);
});

test('round-trips a plain number[]', () => {
  const original = [0.1, 0.2, 0.3];
  const decoded = decodeVector(encodeVector(original));
  assert.ok(decoded);
  assert.deepEqual(decoded.vector, toFloat32(original));
});

test('narrows a Float64Array to float32 precision', () => {
  const original = new Float64Array([1 / 3, 2 / 3]);
  const decoded = decodeVector(encodeVector(original));
  assert.ok(decoded);
  assert.deepEqual(decoded.vector, Float32Array.from(original));
});

test('round-trips a zero-length vector', () => {
  const decoded = decodeVector(encodeVector([]));
  assert.ok(decoded);
  assert.equal(decoded.dim, 0);
  assert.equal(decoded.vector.length, 0);
});

test('encoded buffer size matches header plus payload', () => {
  const buf = encodeVector(new Float32Array(1536));
  assert.equal(buf.length, 16 + 1536 * 4);
});

test('toFloat32 returns the same instance for a Float32Array', () => {
  const original = Float32Array.from([1, 2, 3]);
  assert.equal(toFloat32(original), original);
});

test('toFloat32 rejects anything else', () => {
  assert.throws(() => toFloat32('nope' as any), TypeError);
});

test('rejects a buffer that is too short to hold a header', () => {
  assert.equal(decodeVector(Buffer.alloc(4)), null);
});

test('rejects a bad magic', () => {
  const buf = encodeVector([1, 2, 3]);
  buf.write('XXXX', 0, 'ascii');
  assert.equal(decodeVector(buf), null);
});

test('rejects an unknown format version', () => {
  const buf = encodeVector([1, 2, 3]);
  buf.writeUInt8(99, 4);
  assert.equal(decodeVector(buf), null);
});

test('rejects a length that does not match the declared dim', () => {
  const buf = encodeVector([1, 2, 3]);
  const truncated = buf.subarray(0, buf.length - 4);
  assert.equal(decodeVector(truncated), null);
});

test('rejects a corrupted payload', () => {
  const buf = encodeVector([1, 2, 3]);
  buf[16] ^= 0xff;
  assert.equal(decodeVector(buf), null);
});

test('rejects a vector of the wrong dimension when expectedDim is set', () => {
  const buf = encodeVector([1, 2, 3]);
  assert.equal(decodeVector(buf, 4), null);
  assert.ok(decodeVector(buf, 3));
});
