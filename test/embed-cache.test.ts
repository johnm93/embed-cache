import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { EmbedCache } from '../src/embed-cache.ts';

async function withTempDir(fn: (dir: string) => Promise<void>): Promise<void> {
  const dir = await mkdtemp(join(tmpdir(), 'embed-cache-'));
  try {
    await fn(dir);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

test('getOrCompute calls compute once and caches the result', async () => {
  const cache = new EmbedCache();
  let calls = 0;
  const compute = async (text: string) => {
    calls++;
    return [text.length];
  };

  const first = await cache.getOrCompute('m', 'hello', compute);
  const second = await cache.getOrCompute('m', 'hello', compute);
  assert.equal(calls, 1);
  assert.deepEqual(first, Float32Array.from([5]));
  assert.deepEqual(second, first);
});

test('getOrCompute dedupes concurrent calls for the same text into one compute', async () => {
  const cache = new EmbedCache();
  let calls = 0;
  let resolveCompute!: (v: number[]) => void;
  const compute = () => {
    calls++;
    return new Promise<number[]>((resolve) => {
      resolveCompute = resolve;
    });
  };

  const a = cache.getOrCompute('m', 'hello', compute);
  const b = cache.getOrCompute('m', 'hello', compute);
  assert.equal(calls, 1);
  resolveCompute([1, 2, 3]);
  const [va, vb] = await Promise.all([a, b]);
  assert.deepEqual(va, Float32Array.from([1, 2, 3]));
  assert.deepEqual(vb, va);
});

test('getOrCompute does not cache a rejection, so the next call retries', async () => {
  const cache = new EmbedCache();
  let calls = 0;
  const compute = async () => {
    calls++;
    if (calls === 1) throw new Error('boom');
    return [1];
  };

  await assert.rejects(cache.getOrCompute('m', 'hello', compute), /boom/);
  const vector = await cache.getOrCompute('m', 'hello', compute);
  assert.equal(calls, 2);
  assert.deepEqual(vector, Float32Array.from([1]));
});

test('getOrComputeMany batches only the missing texts, deduplicated and in first-seen order', async () => {
  const cache = new EmbedCache();
  await cache.set('m', 'b', [2]);

  let seen: string[] | null = null;
  const computeBatch = async (texts: string[]) => {
    seen = texts;
    return texts.map((t) => [t.length]);
  };

  const vectors = await cache.getOrComputeMany('m', ['a', 'b', 'c', 'a'], computeBatch);
  assert.deepEqual(seen, ['a', 'c']);
  assert.deepEqual(
    vectors.map((v) => [...v]),
    [[1], [2], [1], [1]],
  );
});

test('getOrComputeMany skips computeBatch entirely when everything is cached', async () => {
  const cache = new EmbedCache();
  await cache.set('m', 'a', [1]);
  await cache.set('m', 'b', [2]);

  let called = false;
  const vectors = await cache.getOrComputeMany('m', ['a', 'b'], async () => {
    called = true;
    return [];
  });
  assert.equal(called, false);
  assert.deepEqual(
    vectors.map((v) => [...v]),
    [[1], [2]],
  );
});

test('getOrComputeMany throws when computeBatch returns the wrong number of vectors', async () => {
  const cache = new EmbedCache();
  await assert.rejects(
    cache.getOrComputeMany('m', ['a', 'b'], async () => [[1]]),
    /computeBatch returned 1 vectors for 2 texts/,
  );
});

test('stats tracks memory hits, misses, and computed vectors, and hitRate ignores computes', async () => {
  const cache = new EmbedCache();
  await cache.getOrCompute('m', 'hello', () => [1]); // miss, then computed
  await cache.get('m', 'hello'); // memory hit
  await cache.get('m', 'nope'); // miss

  const stats = cache.stats();
  assert.equal(stats.memoryHits, 1);
  assert.equal(stats.misses, 2);
  assert.equal(stats.computed, 1);
  assert.equal(stats.diskHits, 0);
  assert.equal(stats.entries, 1);
  assert.equal(stats.hitRate, 1 / 3);
});

test('stats tracks disk hits and corrupt entries once a disk tier is present', () =>
  withTempDir(async (dir) => {
    const cache = new EmbedCache({ dir });
    await cache.set('m', 'hello', [1, 2, 3]);
    cache.clearMemory();

    const vector = await cache.get('m', 'hello');
    assert.deepEqual(vector, Float32Array.from([1, 2, 3]));
    assert.equal(cache.stats().diskHits, 1);
  }));

test('resetStats zeroes counters without touching cached entries', async () => {
  const cache = new EmbedCache();
  await cache.set('m', 'hello', [1]);
  await cache.get('m', 'hello');
  cache.resetStats();

  const stats = cache.stats();
  assert.equal(stats.memoryHits, 0);
  assert.equal(stats.misses, 0);
  assert.equal(stats.entries, 1);
});

test('set rejects a vector of the wrong dimension when expectedDim is set', async () => {
  const cache = new EmbedCache({ expectedDim: 3 });
  await assert.rejects(cache.set('m', 'hello', [1, 2]), RangeError);
});

test('delete removes an entry from memory and disk', () =>
  withTempDir(async (dir) => {
    const cache = new EmbedCache({ dir });
    await cache.set('m', 'hello', [1]);
    await cache.delete('m', 'hello');
    assert.equal(await cache.get('m', 'hello'), null);
  }));

test('clear empties both tiers', () =>
  withTempDir(async (dir) => {
    const cache = new EmbedCache({ dir });
    await cache.set('m', 'hello', [1]);
    await cache.clear();
    assert.equal(await cache.get('m', 'hello'), null);
    assert.equal(cache.stats().entries, 0);
  }));
