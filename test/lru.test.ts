import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Lru } from '../src/lru.ts';

test('get and peek return stored values', () => {
  const lru = new Lru<string, number>();
  lru.set('a', 1);
  assert.equal(lru.get('a'), 1);
  assert.equal(lru.peek('a'), 1);
  assert.equal(lru.get('missing'), undefined);
});

test('has and size reflect current contents', () => {
  const lru = new Lru<string, number>();
  assert.equal(lru.size, 0);
  lru.set('a', 1);
  lru.set('b', 2);
  assert.equal(lru.size, 2);
  assert.ok(lru.has('a'));
  assert.ok(!lru.has('z'));
});

test('delete removes an entry and reports whether it existed', () => {
  const lru = new Lru<string, number>();
  lru.set('a', 1);
  assert.equal(lru.delete('a'), true);
  assert.equal(lru.delete('a'), false);
  assert.equal(lru.has('a'), false);
});

test('clear empties the cache and resets byte tracking', () => {
  const lru = new Lru<string, number>({ sizeOf: () => 10 });
  lru.set('a', 1);
  lru.set('b', 2);
  lru.clear();
  assert.equal(lru.size, 0);
  assert.equal(lru.bytes, 0);
});

test('evicts least-recently-used entry once maxEntries is exceeded', () => {
  const lru = new Lru<string, number>({ maxEntries: 2 });
  lru.set('a', 1);
  lru.set('b', 2);
  lru.set('c', 3);
  assert.deepEqual([...lru.keys()], ['b', 'c']);
  assert.equal(lru.has('a'), false);
});

test('get promotes an entry so it survives eviction', () => {
  const lru = new Lru<string, number>({ maxEntries: 2 });
  lru.set('a', 1);
  lru.set('b', 2);
  lru.get('a'); // a is now most-recently-used
  lru.set('c', 3);
  assert.deepEqual([...lru.keys()], ['a', 'c']);
  assert.equal(lru.has('b'), false);
});

test('peek does not affect eviction order', () => {
  const lru = new Lru<string, number>({ maxEntries: 2 });
  lru.set('a', 1);
  lru.set('b', 2);
  lru.peek('a');
  lru.set('c', 3);
  assert.deepEqual([...lru.keys()], ['b', 'c']);
});

test('re-setting an existing key updates its value and recency without double counting bytes', () => {
  const lru = new Lru<string, string>({ maxBytes: 100, sizeOf: (v) => v.length });
  lru.set('a', 'xxxxx'); // 5 bytes
  lru.set('a', 'xx'); // 2 bytes
  assert.equal(lru.bytes, 2);
  assert.equal(lru.get('a'), 'xx');
});

test('evicts oldest entries once maxBytes is exceeded', () => {
  const lru = new Lru<string, string>({ maxBytes: 10, sizeOf: (v) => v.length });
  lru.set('a', 'aaaa'); // 4
  lru.set('b', 'bbbb'); // 4, total 8
  lru.set('c', 'cc'); // 2, total 10 -- still fits
  assert.deepEqual([...lru.keys()], ['a', 'b', 'c']);
  lru.set('d', 'd'); // 1, total 11 -- evict 'a' (4) to fit
  assert.deepEqual([...lru.keys()], ['b', 'c', 'd']);
  assert.equal(lru.bytes, 7);
});

test('keeps a single oversized entry rather than evicting to empty', () => {
  const lru = new Lru<string, string>({ maxBytes: 5, sizeOf: (v) => v.length });
  lru.set('huge', 'this value is much bigger than the budget');
  assert.equal(lru.size, 1);
  assert.ok(lru.has('huge'));
});

test('an oversized entry is evicted once something else is added', () => {
  const lru = new Lru<string, string>({ maxBytes: 5, sizeOf: (v) => v.length });
  lru.set('huge', 'way over budget on its own');
  lru.set('small', 'ok'); // 2 bytes, fits once 'huge' is gone
  assert.deepEqual([...lru.keys()], ['small']);
});

test('rejects a non-positive maxEntries or maxBytes', () => {
  assert.throws(() => new Lru({ maxEntries: 0 }), RangeError);
  assert.throws(() => new Lru({ maxBytes: -1 }), RangeError);
});
