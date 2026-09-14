import { test } from 'node:test';
import assert from 'node:assert/strict';
import { embedKey, isCacheKey, assertCacheKey, keyToSegments } from '../src/key.ts';

test('is deterministic for the same inputs', () => {
  assert.equal(embedKey('model-a', 'hello'), embedKey('model-a', 'hello'));
});

test('is a 64-character lowercase hex string', () => {
  assert.match(embedKey('model-a', 'hello'), /^[0-9a-f]{64}$/);
});

test('differs when the model changes', () => {
  assert.notEqual(embedKey('model-a', 'hello'), embedKey('model-b', 'hello'));
});

test('differs when the text changes', () => {
  assert.notEqual(embedKey('model-a', 'hello'), embedKey('model-a', 'goodbye'));
});

test('differs when the namespace changes', () => {
  assert.notEqual(embedKey('model-a', 'hello', 'ns1'), embedKey('model-a', 'hello', 'ns2'));
});

test('defaults to an empty namespace', () => {
  assert.equal(embedKey('model-a', 'hello'), embedKey('model-a', 'hello', ''));
});

test('is case- and whitespace-sensitive on text', () => {
  assert.notEqual(embedKey('model-a', 'Hello'), embedKey('model-a', 'hello'));
  assert.notEqual(embedKey('model-a', 'hello '), embedKey('model-a', 'hello'));
});

test('framing prevents field-boundary collisions', () => {
  // Without a length prefix, model="a"+text="bc" would collide with model="ab"+text="c".
  assert.notEqual(embedKey('a', 'bc'), embedKey('ab', 'c'));
  // Same idea across the namespace/model boundary.
  assert.notEqual(embedKey('b', 'x', 'a'), embedKey('ab', 'x', ''));
});

test('rejects a non-string or empty model', () => {
  assert.throws(() => embedKey('', 'hello'), TypeError);
  assert.throws(() => embedKey(null as any, 'hello'), TypeError);
});

test('rejects a non-string text', () => {
  assert.throws(() => embedKey('model-a', null as any), TypeError);
  assert.throws(() => embedKey('model-a', 5 as any), TypeError);
});

test('rejects a non-string namespace', () => {
  assert.throws(() => embedKey('model-a', 'hello', 5 as any), TypeError);
});

test('isCacheKey accepts a well-formed key and rejects everything else', () => {
  const key = embedKey('model-a', 'hello');
  assert.equal(isCacheKey(key), true);
  assert.equal(isCacheKey(key.toUpperCase()), false);
  assert.equal(isCacheKey(key.slice(0, -1)), false);
  assert.equal(isCacheKey(`${key}0`), false);
  assert.equal(isCacheKey('not-a-key'), false);
  assert.equal(isCacheKey(''), false);
});

test('assertCacheKey throws on an invalid key and returns on a valid one', () => {
  const key = embedKey('model-a', 'hello');
  assert.doesNotThrow(() => assertCacheKey(key));
  assert.throws(() => assertCacheKey('nope'), TypeError);
});

test('keyToSegments splits a key into a two-level shard path plus filename', () => {
  const key = embedKey('model-a', 'hello');
  const segments = keyToSegments(key);
  assert.equal(segments.length, 3);
  assert.equal(segments[0], key.slice(0, 2));
  assert.equal(segments[1], key.slice(2, 4));
  assert.equal(segments[2], `${key.slice(4)}.vec`);
});

test('keyToSegments rejects a key that is not a valid cache key', () => {
  assert.throws(() => keyToSegments('not-a-key'), TypeError);
});
