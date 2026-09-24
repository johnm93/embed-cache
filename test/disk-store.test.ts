import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DiskStore } from '../src/disk-store.ts';
import { embedKey, keyToSegments } from '../src/key.ts';

async function withTempDir(fn: (dir: string) => Promise<void>): Promise<void> {
  const dir = await mkdtemp(join(tmpdir(), 'embed-cache-'));
  try {
    await fn(dir);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

test('rejects an empty dir', () => {
  assert.throws(() => new DiskStore(''), TypeError);
});

test('set then get round-trips a vector', () =>
  withTempDir(async (dir) => {
    const store = new DiskStore(dir);
    const key = embedKey('m', 'hello');
    await store.set(key, [1, 2, 3]);
    const result = await store.get(key);
    assert.equal(result.corrupt, false);
    assert.deepEqual(result.vector, Float32Array.from([1, 2, 3]));
  }));

test('get on a missing key is a miss, not corrupt', () =>
  withTempDir(async (dir) => {
    const store = new DiskStore(dir);
    const result = await store.get(embedKey('m', 'nope'));
    assert.equal(result.vector, null);
    assert.equal(result.corrupt, false);
  }));

test('writes are sharded under the first four hex characters of the key', () =>
  withTempDir(async (dir) => {
    const store = new DiskStore(dir);
    const key = embedKey('m', 'hello');
    await store.set(key, [1]);
    const [a, b, file] = keyToSegments(key);
    const buf = await readFile(join(dir, a, b, file));
    assert.equal(buf.subarray(0, 4).toString('ascii'), 'EMBC');
  }));

test('delete removes a stored vector and reports whether it existed', () =>
  withTempDir(async (dir) => {
    const store = new DiskStore(dir);
    const key = embedKey('m', 'hello');
    await store.set(key, [1]);
    assert.equal(await store.delete(key), true);
    assert.equal(await store.delete(key), false);
    assert.equal((await store.get(key)).vector, null);
  }));

test('set overwrites an existing entry', () =>
  withTempDir(async (dir) => {
    const store = new DiskStore(dir);
    const key = embedKey('m', 'hello');
    await store.set(key, [1, 2, 3]);
    await store.set(key, [4, 5]);
    const result = await store.get(key);
    assert.deepEqual(result.vector, Float32Array.from([4, 5]));
  }));

test('a corrupted file is reported as corrupt and then removed', () =>
  withTempDir(async (dir) => {
    const store = new DiskStore(dir);
    const key = embedKey('m', 'hello');
    await store.set(key, [1, 2, 3]);

    const path = join(dir, ...keyToSegments(key));
    const buf = await readFile(path);
    buf[16] ^= 0xff; // flip a payload bit so the crc no longer matches
    await writeFile(path, buf);

    const first = await store.get(key);
    assert.equal(first.vector, null);
    assert.equal(first.corrupt, true);

    const second = await store.get(key);
    assert.equal(second.vector, null);
    assert.equal(second.corrupt, false); // the bad file was deleted, so now it's just a miss
  }));

test('a vector of the wrong dimension is treated as corrupt for this store', () =>
  withTempDir(async (dir) => {
    const store = new DiskStore(dir);
    const key = embedKey('m', 'hello');
    await store.set(key, [1, 2, 3]);
    const result = await store.get(key, 4);
    assert.equal(result.vector, null);
    assert.equal(result.corrupt, true);
  }));

test('clear removes everything but leaves the store usable', () =>
  withTempDir(async (dir) => {
    const store = new DiskStore(dir);
    const key = embedKey('m', 'hello');
    await store.set(key, [1]);
    await store.clear();
    assert.equal((await store.get(key)).vector, null);

    await store.set(key, [2]);
    assert.deepEqual((await store.get(key)).vector, Float32Array.from([2]));
  }));

test('does not leave temp files behind after a successful write', () =>
  withTempDir(async (dir) => {
    const store = new DiskStore(dir);
    const key = embedKey('m', 'hello');
    await store.set(key, [1]);
    const [a, b] = keyToSegments(key);
    const names = await readdir(join(dir, a, b));
    assert.deepEqual(names, [`${key.slice(4)}.vec`]);
  }));

test('a zero-byte file is reported as corrupt and then removed', () =>
  withTempDir(async (dir) => {
    const store = new DiskStore(dir);
    const key = embedKey('m', 'hello');
    const [a, b, file] = keyToSegments(key);
    await mkdir(join(dir, a, b), { recursive: true });
    await writeFile(join(dir, a, b, file), Buffer.alloc(0));

    const result = await store.get(key);
    assert.equal(result.vector, null);
    assert.equal(result.corrupt, true);
    assert.equal((await store.get(key)).corrupt, false); // gone after the first read
  }));

test('a leftover temp file from an interrupted write is ignored', () =>
  withTempDir(async (dir) => {
    const store = new DiskStore(dir);
    const key = embedKey('m', 'hello');
    await store.set(key, [1, 2, 3]);

    const [a, b] = keyToSegments(key);
    const shard = join(dir, a, b);
    await writeFile(join(shard, `.${key}.deadbeef.tmp`), Buffer.alloc(4));

    const result = await store.get(key);
    assert.deepEqual(result.vector, Float32Array.from([1, 2, 3]));
    assert.equal(result.corrupt, false);
  }));

test('two keys sharing a shard directory do not collide', () =>
  withTempDir(async (dir) => {
    const store = new DiskStore(dir);
    // synthetic keys, chosen only to share their first four hex characters
    // so both land in the same shard directory.
    const keyA = '0'.repeat(64);
    const keyB = `0000${'f'.repeat(60)}`;
    await store.set(keyA, [1, 2]);
    await store.set(keyB, [3, 4, 5]);
    assert.deepEqual((await store.get(keyA)).vector, Float32Array.from([1, 2]));
    assert.deepEqual((await store.get(keyB)).vector, Float32Array.from([3, 4, 5]));
  }));

test('an explicit null expectedDim skips the dim check', () =>
  withTempDir(async (dir) => {
    const store = new DiskStore(dir);
    const key = embedKey('m', 'hello');
    await store.set(key, [1, 2, 3]);
    const result = await store.get(key, null);
    assert.deepEqual(result.vector, Float32Array.from([1, 2, 3]));
    assert.equal(result.corrupt, false);
  }));
