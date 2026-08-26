import { toFloat32 } from './codec.js';
import { DiskStore } from './disk-store.js';
import { embedKey } from './key.js';
import { Lru } from './lru.js';
import type { CacheStats, ComputeMany, ComputeOne, EmbedCacheOptions, VectorInput } from './types.js';

interface RawStats {
  memoryHits: number;
  diskHits: number;
  misses: number;
  computed: number;
  corrupt: number;
  diskWrites: number;
}

function freshStats(): RawStats {
  return { memoryHits: 0, diskHits: 0, misses: 0, computed: 0, corrupt: 0, diskWrites: 0 };
}

/**
 * LRU-in-front-of-disk cache keyed by (model, text). See README for the
 * on-disk format and the reasoning behind it.
 */
export class EmbedCache {
  private readonly memory: Lru<string, Float32Array>;
  private readonly disk: DiskStore | null;
  private readonly namespace: string;
  private readonly expectedDim: number | null;
  /**
   * One entry per key currently being computed. This is what makes fifty
   * concurrent getOrCompute calls for the same text produce one API call:
   * the second caller finds this map already populated and awaits the same
   * promise instead of starting its own compute.
   */
  private readonly inFlight = new Map<string, Promise<Float32Array>>();
  private rawStats: RawStats = freshStats();

  constructor(options: EmbedCacheOptions = {}) {
    const { dir = null, maxEntries, maxBytes, namespace = '', expectedDim = null } = options;
    this.memory = new Lru<string, Float32Array>({ maxEntries, maxBytes, sizeOf: (v) => v.byteLength });
    this.disk = dir != null ? new DiskStore(dir) : null;
    this.namespace = namespace;
    this.expectedDim = expectedDim ?? null;
  }

  async get(model: string, text: string): Promise<Float32Array | null> {
    return this.getByKey(embedKey(model, text, this.namespace));
  }

  async set(model: string, text: string, input: VectorInput): Promise<void> {
    const vector = this.checkDim(toFloat32(input));
    await this.setByKey(embedKey(model, text, this.namespace), vector);
  }

  async delete(model: string, text: string): Promise<void> {
    const key = embedKey(model, text, this.namespace);
    this.memory.delete(key);
    if (this.disk) await this.disk.delete(key);
  }

  /**
   * Cached vector, or compute it once and store it. Rejections are not
   * cached: a failed compute clears the in-flight entry so the next call
   * tries again instead of replaying the same error forever.
   */
  async getOrCompute(model: string, text: string, compute: ComputeOne): Promise<Float32Array> {
    const key = embedKey(model, text, this.namespace);
    const existing = this.inFlight.get(key);
    if (existing) return existing;

    const promise = this.computeAndStore(key, model, text, compute).finally(() => {
      if (this.inFlight.get(key) === promise) this.inFlight.delete(key);
    });
    this.inFlight.set(key, promise);
    return promise;
  }

  /**
   * As getOrCompute for an array. Everything is looked up first; computeBatch
   * is called at most once, with only the texts that were missing,
   * deduplicated and in first-seen order. The result lines up positionally
   * with `texts`, including duplicates.
   */
  async getOrComputeMany(model: string, texts: string[], computeBatch: ComputeMany): Promise<Float32Array[]> {
    const keys = texts.map((text) => embedKey(model, text, this.namespace));
    const results: (Float32Array | undefined)[] = new Array(texts.length);
    const missingKeys: string[] = [];
    const missingTexts: string[] = [];
    const missingIndexForKey = new Map<string, number>();

    for (let i = 0; i < texts.length; i++) {
      const key = keys[i];
      const cached = await this.getByKey(key);
      if (cached) {
        results[i] = cached;
        continue;
      }
      if (!missingIndexForKey.has(key)) {
        missingIndexForKey.set(key, missingKeys.length);
        missingKeys.push(key);
        missingTexts.push(texts[i]!);
      }
    }

    if (missingTexts.length > 0) {
      const raw = await computeBatch(missingTexts, model);
      if (raw.length !== missingTexts.length) {
        throw new Error(`computeBatch returned ${raw.length} vectors for ${missingTexts.length} texts`);
      }
      const vectors = raw.map((v) => this.checkDim(toFloat32(v)));
      for (let i = 0; i < vectors.length; i++) {
        await this.setByKey(missingKeys[i]!, vectors[i]!);
      }
      this.rawStats.computed += vectors.length;

      for (let i = 0; i < texts.length; i++) {
        if (results[i] === undefined) {
          const idx = missingIndexForKey.get(keys[i]!);
          if (idx !== undefined) results[i] = vectors[idx];
        }
      }
    }

    return results as Float32Array[];
  }

  stats(): CacheStats {
    const { memoryHits, diskHits, misses, computed, corrupt, diskWrites } = this.rawStats;
    const lookups = memoryHits + diskHits + misses;
    return {
      memoryHits,
      diskHits,
      misses,
      computed,
      corrupt,
      diskWrites,
      entries: this.memory.size,
      bytes: this.memory.bytes,
      hitRate: lookups === 0 ? 0 : (memoryHits + diskHits) / lookups,
    };
  }

  resetStats(): void {
    this.rawStats = freshStats();
  }

  clearMemory(): void {
    this.memory.clear();
  }

  async clear(): Promise<void> {
    this.memory.clear();
    if (this.disk) await this.disk.clear();
  }

  private async computeAndStore(key: string, model: string, text: string, compute: ComputeOne): Promise<Float32Array> {
    const cached = await this.getByKey(key);
    if (cached) return cached;
    const vector = this.checkDim(toFloat32(await compute(text, model)));
    await this.setByKey(key, vector);
    this.rawStats.computed++;
    return vector;
  }

  private async getByKey(key: string): Promise<Float32Array | null> {
    const mem = this.memory.get(key);
    if (mem) {
      this.rawStats.memoryHits++;
      return mem;
    }
    if (this.disk) {
      const result = await this.disk.get(key, this.expectedDim);
      if (result.corrupt) this.rawStats.corrupt++;
      if (result.vector) {
        this.memory.set(key, result.vector);
        this.rawStats.diskHits++;
        return result.vector;
      }
    }
    this.rawStats.misses++;
    return null;
  }

  private async setByKey(key: string, vector: Float32Array): Promise<void> {
    this.memory.set(key, vector);
    if (this.disk) {
      await this.disk.set(key, vector);
      this.rawStats.diskWrites++;
    }
  }

  private checkDim(vector: Float32Array): Float32Array {
    if (this.expectedDim != null && vector.length !== this.expectedDim) {
      throw new RangeError(`expected a ${this.expectedDim}-dimension vector, got ${vector.length}`);
    }
    return vector;
  }
}
